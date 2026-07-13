from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    AGENT_ERROR = "AE"
    TRANSIENT_FAILURE = "TF"
    PERSISTENT_DRIFT = "PD"


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    target_tool: str
    drift_type: str
    active: bool
    persistent: bool
    activation_episode: int
    deactivation_episode: int | None
    mutation_operation: str
    mutation: dict[str, Any]
    input_adapter: str | None = None
    response_adapter: str | None = None
    workflow_guard: str | None = None
    state_effect_strategy: str | None = None

    def active_at(self, episode_index: int) -> bool:
        if episode_index < self.activation_episode:
            return False
        return self.deactivation_episode is None or episode_index <= self.deactivation_episode


@dataclass(frozen=True)
class ExecutionProfile:
    mode: ExecutionMode
    drift_case: dict[str, Any]
    family: dict[str, Any]
    active_episode: int = 3
    change_point: int = 3
    candidate_patch_active: bool = False

    @property
    def scenario_id(self) -> str:
        return f"{self.family['matched_case_id']}-{self.mode.value}"

    @property
    def target_tool(self) -> str:
        return str(self.drift_case["target_tool"])

    @property
    def runtime_profile(self) -> RuntimeProfile:
        drift_type = str(self.drift_case["drift_type"])
        operation = str(self.drift_case["runtime_mutation"]["operation"])
        if self.mode == ExecutionMode.AGENT_ERROR:
            activation, deactivation = 1, None
        else:
            activation = self.change_point
            deactivation = None if self.mode == ExecutionMode.PERSISTENT_DRIFT else self.active_episode
        return RuntimeProfile(
            profile_id=f"profile-{self.family['matched_case_id']}",
            target_tool=self.target_tool,
            drift_type=drift_type,
            active=not self.candidate_patch_active,
            persistent=self.mode == ExecutionMode.PERSISTENT_DRIFT,
            activation_episode=activation,
            deactivation_episode=deactivation,
            mutation_operation=operation,
            mutation=dict(self.drift_case["runtime_mutation"]),
            input_adapter=operation if drift_type == "input_contract" else None,
            response_adapter=operation if drift_type == "response_shape" else None,
            workflow_guard=operation if drift_type == "workflow_precondition" else None,
            state_effect_strategy=operation if drift_type == "state_effect" else None,
        )

    def scheduled(self, episode_index: int) -> bool:
        if self.candidate_patch_active:
            return False
        if self.mode in {ExecutionMode.AGENT_ERROR, ExecutionMode.TRANSIENT_FAILURE}:
            return episode_index == self.active_episode
        return episode_index >= self.change_point

    def runtime_contract_active(self, episode_index: int) -> bool:
        if self.candidate_patch_active:
            return False
        if self.mode == ExecutionMode.AGENT_ERROR:
            return True
        if self.mode == ExecutionMode.PERSISTENT_DRIFT:
            return episode_index >= self.change_point
        return False
