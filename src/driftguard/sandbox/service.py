from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from driftguard.contracts.input_validator import InputValidationError, InputValidator
from driftguard.contracts.registry import ContractRegistry, ToolContract
from driftguard.sandbox.deterministic_clock import DeterministicClock
from driftguard.sandbox.diff import state_diff
from driftguard.sandbox.errors import SandboxError
from driftguard.sandbox.permissions import actor_has_permission
from driftguard.sandbox.result import ToolResult, failure, success
from driftguard.sandbox.state_store import StateStore
from driftguard.sandbox.tools import HANDLERS


@dataclass(frozen=True)
class CallRecord:
    call_index: int
    tool: str
    arguments: dict[str, Any]
    actor_id: str
    status_code: int
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    state_diff: list[dict[str, Any]]
    clock_before: str
    clock_after: str


class SandboxService:
    def __init__(
        self,
        store: StateStore | None = None,
        registry: ContractRegistry | None = None,
        validator: InputValidator | None = None,
    ):
        self.store = store or StateStore.from_fixture()
        self.registry = registry or ContractRegistry.from_openapi()
        self.validator = validator or InputValidator()
        self.clock = DeterministicClock(self.store.initial_clock)
        self._records: list[CallRecord] = []
        if set(self.registry.operation_ids()) != set(HANDLERS):
            raise ValueError("handler set must exactly match canonical OpenAPI operations")

    def reset(self) -> None:
        self.store.reset()
        self.clock.reset()
        self._records = []

    def call_log(self) -> list[dict[str, Any]]:
        return [deepcopy(asdict(record)) for record in self._records]

    @staticmethod
    def _project_output(contract: ToolContract, data: dict[str, Any]) -> dict[str, Any]:
        data_schema = contract.success_schema["properties"]["data"]
        fields = data_schema.get("properties", {})
        return {name: deepcopy(data[name]) for name in fields if name in data}

    def _record(
        self,
        tool: str,
        arguments: dict[str, Any],
        actor_id: str,
        result: ToolResult,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        clock_before: str,
    ) -> None:
        self._records.append(
            CallRecord(
                call_index=len(self._records) + 1,
                tool=tool,
                arguments=deepcopy(arguments),
                actor_id=actor_id,
                status_code=result.status_code,
                pre_state=deepcopy(pre_state),
                post_state=deepcopy(post_state),
                state_diff=state_diff(pre_state, post_state),
                clock_before=clock_before,
                clock_after=self.clock.now(),
            )
        )

    def call_tool(self, tool: str, arguments: dict[str, Any], actor_id: str) -> ToolResult:
        pre_state = self.store.snapshot()
        clock_before = self.clock.now()
        contract = self.registry.get(tool)
        if contract is None:
            result = failure(400, "BAD_REQUEST", f"Unknown tool: {tool}", "tool")
            self._record(tool, arguments, actor_id, result, pre_state, pre_state, clock_before)
            return result
        try:
            normalized = self.validator.validate(contract, arguments)
        except InputValidationError as exc:
            result = failure(422, "VALIDATION_ERROR", str(exc), exc.field)
            self._record(tool, arguments, actor_id, result, pre_state, pre_state, clock_before)
            return result
        if not actor_has_permission(pre_state, actor_id, contract.required_permission):
            result = failure(403, "PERMISSION_DENIED", "Actor lacks the required permission.")
            self._record(tool, normalized, actor_id, result, pre_state, pre_state, clock_before)
            return result

        working_state = deepcopy(pre_state)
        timestamp = self.clock.peek_next() if contract.effect_type == "write" else None
        try:
            data = HANDLERS[tool](working_state, normalized, timestamp)
        except SandboxError as exc:
            result = failure(exc.status_code, exc.code, str(exc), exc.field)
            self._record(tool, normalized, actor_id, result, pre_state, pre_state, clock_before)
            return result

        if contract.effect_type == "write":
            self.store.commit(working_state)
            advanced = self.clock.advance_write()
            if advanced != timestamp:
                raise RuntimeError("deterministic clock mismatch")
        post_state = self.store.snapshot()
        result = success(contract.success_status_code, self._project_output(contract, data))
        self._record(tool, normalized, actor_id, result, pre_state, post_state, clock_before)
        return result
