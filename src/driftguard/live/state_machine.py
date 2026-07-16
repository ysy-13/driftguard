from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class LiveHealingState(str, Enum):
    TASK_EXECUTION = "TASK_EXECUTION"
    FAILURE_OBSERVED = "FAILURE_OBSERVED"
    PRELIMINARY_ATTRIBUTION = "PRELIMINARY_ATTRIBUTION"
    RETRY_OR_EVIDENCE_COLLECTION = "RETRY_OR_EVIDENCE_COLLECTION"
    REPRODUCTION_CHECK = "REPRODUCTION_CHECK"
    PROBE_SELECTION = "PROBE_SELECTION"
    PROBE_EXECUTION = "PROBE_EXECUTION"
    FINAL_ATTRIBUTION = "FINAL_ATTRIBUTION"
    PATCH_ELIGIBILITY = "PATCH_ELIGIBILITY"
    PATCH_PROPOSAL = "PATCH_PROPOSAL"
    PATCH_VALIDATION = "PATCH_VALIDATION"
    IMMEDIATE_REPAIR = "IMMEDIATE_REPAIR"
    PATCH_ACCEPTED = "PATCH_ACCEPTED"
    PATCH_REJECTED = "PATCH_REJECTED"
    FUTURE_TRANSFER = "FUTURE_TRANSFER"
    FINAL_EVALUATION = "FINAL_EVALUATION"


ORDER = tuple(LiveHealingState)


@dataclass(frozen=True)
class LiveStateTransition:
    previous_state: str | None
    new_state: str
    reason_code: str
    evidence_refs: tuple[str, ...]
    llm_call_id: str | None
    operation_call_id: str | None
    remaining_budget: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


class LiveHealingStateMachine:
    def __init__(self) -> None:
        self.state: LiveHealingState | None = None
        self.transitions: list[LiveStateTransition] = []

    def move(
        self, new_state: LiveHealingState | str, reason_code: str, *,
        evidence_refs: tuple[str, ...] = (), llm_call_id: str | None = None,
        operation_call_id: str | None = None, remaining_budget: dict[str, Any] | None = None,
    ) -> None:
        new = LiveHealingState(new_state)
        if self.state is not None:
            current_index, new_index = ORDER.index(self.state), ORDER.index(new)
            if new_index < current_index and not (
                self.state == LiveHealingState.PATCH_REJECTED and new == LiveHealingState.FINAL_EVALUATION
            ):
                raise ValueError(f"invalid backward live-healing transition: {self.state.value}->{new.value}")
            if self.state == LiveHealingState.PATCH_ELIGIBILITY and new == LiveHealingState.PATCH_VALIDATION:
                raise ValueError("Patch Eligibility cannot be bypassed")
        self.transitions.append(LiveStateTransition(
            self.state.value if self.state else None, new.value, reason_code,
            tuple(evidence_refs), llm_call_id, operation_call_id, dict(remaining_budget or {}),
        ))
        self.state = new

    def visible(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.transitions]
