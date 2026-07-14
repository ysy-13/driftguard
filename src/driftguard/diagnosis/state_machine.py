from __future__ import annotations

from .models import DiagnosisState, StateTransition


class DiagnosisStateMachine:
    def __init__(self):
        self.state = DiagnosisState.UNRESOLVED
        self.transitions: list[StateTransition] = []

    def transition(
        self,
        new_state: DiagnosisState,
        evidence_ids: tuple[str, ...],
        rejected: tuple[str, ...],
        reason_code: str,
        confidence: float,
    ) -> None:
        transition = StateTransition(
            self.state.value, new_state.value, evidence_ids, rejected, reason_code, confidence
        )
        self.transitions.append(transition)
        self.state = new_state
