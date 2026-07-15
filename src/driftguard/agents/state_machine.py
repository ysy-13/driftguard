from __future__ import annotations

from dataclasses import dataclass


TERMINAL_STATES = {
    "SUCCESS", "ABSTAINED", "SAFETY_BLOCKED", "BUDGET_EXHAUSTED",
    "PROVIDER_FAILURE", "INVALID_STRUCTURED_OUTPUT",
}


@dataclass(frozen=True)
class StateTransition:
    sequence: int
    from_state: str | None
    to_state: str
    reason_code: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason_code": self.reason_code,
        }


class ControllerStateMachine:
    def __init__(self) -> None:
        self.state: str | None = None
        self.transitions: list[StateTransition] = []

    def move(self, state: str, reason_code: str) -> None:
        if self.state in TERMINAL_STATES:
            raise RuntimeError(f"cannot transition from terminal state {self.state}")
        self.transitions.append(StateTransition(len(self.transitions) + 1, self.state, state, reason_code))
        self.state = state

    def visible(self) -> tuple[dict[str, object], ...]:
        return tuple(item.to_dict() for item in self.transitions)
