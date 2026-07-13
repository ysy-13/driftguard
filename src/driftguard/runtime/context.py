from __future__ import annotations

from dataclasses import dataclass, field

from .contract_snapshots import ContractSnapshots
from .execution_profile import ExecutionMode, ExecutionProfile
from .session_state import RuntimeSessionState


@dataclass
class ExecutionContext:
    profile: ExecutionProfile
    episode_index: int = 3
    public_scenario_id: str = ""
    task_instance_id: str = ""
    actor_id: str = "agent_admin"
    session_state: RuntimeSessionState = field(default_factory=RuntimeSessionState)

    def __post_init__(self) -> None:
        if not self.public_scenario_id:
            self.public_scenario_id = f"public-{self.profile.family['matched_case_id']}"
        if not self.task_instance_id:
            self.task_instance_id = f"{self.profile.drift_case['drift_id']}-D1"
        self.contracts = ContractSnapshots.for_profile(self.profile, self.episode_index)

    @property
    def matched_case_id(self) -> str:
        return str(self.profile.family["matched_case_id"])

    @property
    def variant(self) -> str:
        return self.profile.mode.value

    @property
    def runtime_profile(self):
        return self.profile.runtime_profile

    @property
    def displayed_contract(self):
        return self.contracts.displayed.copy_document()

    @property
    def runtime_contract(self):
        return self.contracts.runtime.copy_document()

    @property
    def session(self) -> RuntimeSessionState:
        return self.session_state

    @property
    def pending_effects(self):
        return self.session_state.pending_effects

    @property
    def session_key(self) -> str:
        return self.profile.scenario_id

    def set_episode(self, index: int) -> None:
        self.episode_index = index
        self.contracts = ContractSnapshots.for_profile(self.profile, self.episode_index)

    def injection_active(self, tool: str) -> bool:
        if tool != self.profile.target_tool:
            return False
        if self.profile.runtime_contract_active(self.episode_index):
            return True
        if not self.profile.scheduled(self.episode_index):
            return False
        if self.profile.mode == ExecutionMode.TRANSIENT_FAILURE:
            return self.profile.scenario_id not in self.session.consumed_transient
        return False

    @property
    def agent_fault_active(self) -> bool:
        return self.profile.mode == ExecutionMode.AGENT_ERROR and self.episode_index == self.profile.active_episode

    def consume(self) -> None:
        if self.profile.mode == ExecutionMode.TRANSIENT_FAILURE:
            self.session.consumed_transient.add(self.profile.scenario_id)

    def reset_runtime(self) -> None:
        self.session.reset()
        self.contracts = ContractSnapshots.for_profile(self.profile, self.episode_index)

    def reset(self) -> None:
        self.reset_runtime()
