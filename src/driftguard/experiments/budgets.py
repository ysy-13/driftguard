from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

from driftguard.llm.errors import ErrorCategory


class BudgetExhausted(RuntimeError):
    category = ErrorCategory.BUDGET_EXHAUSTED


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: int
    patch_proposal: bool


@dataclass(frozen=True)
class ExperimentBudget:
    max_llm_calls: int = 6
    max_tool_calls: int = 8
    max_probe_calls: int = 3
    max_total_interactions: int = 12
    max_input_tokens: int = 12000
    max_output_tokens: int = 3000
    max_wall_time_seconds: float = 120.0
    max_format_repairs: int = 2
    max_patch_proposal_calls: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BudgetTracker:
    budget: ExperimentBudget
    llm_calls: int = 0
    tool_calls: int = 0
    probe_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    format_repairs: int = 0
    patch_proposal_calls: int = 0
    _started: float = field(default_factory=time.monotonic)
    _next_reservation_id: int = 1
    _reservations: dict[int, BudgetReservation] = field(default_factory=dict)

    @property
    def total_interactions(self) -> int:
        return self.llm_calls + self.tool_calls

    def _check_values(self, llm_calls: int, tool_calls: int, probe_calls: int, input_tokens: int, output_tokens: int, format_repairs: int, patch_proposal_calls: int | None = None) -> None:
        if llm_calls > self.budget.max_llm_calls:
            raise BudgetExhausted("LLM call budget exhausted")
        if tool_calls > self.budget.max_tool_calls:
            raise BudgetExhausted("tool call budget exhausted")
        if probe_calls > self.budget.max_probe_calls:
            raise BudgetExhausted("probe budget exhausted")
        if llm_calls + tool_calls > self.budget.max_total_interactions:
            raise BudgetExhausted("total interaction budget exhausted")
        if input_tokens > self.budget.max_input_tokens or output_tokens > self.budget.max_output_tokens:
            raise BudgetExhausted("token budget exhausted")
        if format_repairs > self.budget.max_format_repairs:
            raise BudgetExhausted("format repair budget exhausted")
        if (self.patch_proposal_calls if patch_proposal_calls is None else patch_proposal_calls) > self.budget.max_patch_proposal_calls:
            raise BudgetExhausted("patch proposal budget exhausted")
        if time.monotonic() - self._started > self.budget.max_wall_time_seconds:
            raise BudgetExhausted("wall-time budget exhausted")

    def consume_llm(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._check_values(
            self.llm_calls + 1, self.tool_calls, self.probe_calls,
            self.input_tokens + input_tokens, self.output_tokens + output_tokens, self.format_repairs,
        )
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def consume_tool(self, probe: bool = False) -> None:
        self._check_values(
            self.llm_calls, self.tool_calls + 1, self.probe_calls + int(probe),
            self.input_tokens, self.output_tokens, self.format_repairs,
        )
        self.tool_calls += 1
        if probe:
            self.probe_calls += 1

    def consume_format_repair(self) -> None:
        self._check_values(
            self.llm_calls, self.tool_calls, self.probe_calls,
            self.input_tokens, self.output_tokens, self.format_repairs + 1,
        )
        self.format_repairs += 1

    def ensure_llm_call_allowed(self) -> None:
        self._check_values(
            self.llm_calls + 1, self.tool_calls, self.probe_calls,
            self.input_tokens, self.output_tokens, self.format_repairs,
        )

    def reserve_llm_call(self, *, patch_proposal: bool = False) -> BudgetReservation:
        """Atomically consumes the interaction slot before a Provider boundary call."""
        next_patch = self.patch_proposal_calls + int(patch_proposal)
        self._check_values(
            self.llm_calls + 1, self.tool_calls, self.probe_calls,
            self.input_tokens, self.output_tokens, self.format_repairs, next_patch,
        )
        reservation = BudgetReservation(self._next_reservation_id, patch_proposal)
        self._next_reservation_id += 1
        self._reservations[reservation.reservation_id] = reservation
        self.llm_calls += 1
        if patch_proposal:
            self.patch_proposal_calls += 1
        return reservation

    def settle_llm_call(
        self, reservation: BudgetReservation, input_tokens: int = 0, output_tokens: int = 0,
    ) -> None:
        if self._reservations.pop(reservation.reservation_id, None) is None:
            raise ValueError("unknown or already settled budget reservation")
        # The response is preserved even when reported tokens cross a method
        # budget.  The next call fails closed during reservation.
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def release_llm_call(self, reservation: BudgetReservation) -> None:
        if self._reservations.pop(reservation.reservation_id, None) is None:
            return
        self.llm_calls -= 1
        if reservation.patch_proposal:
            self.patch_proposal_calls -= 1

    def assert_settled_within_budget(self) -> None:
        if self.input_tokens > self.budget.max_input_tokens or self.output_tokens > self.budget.max_output_tokens:
            raise BudgetExhausted("token budget exhausted")

    def consume_patch_proposal(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._check_values(
            self.llm_calls + 1, self.tool_calls, self.probe_calls,
            self.input_tokens + input_tokens, self.output_tokens + output_tokens,
            self.format_repairs, self.patch_proposal_calls + 1,
        )
        self.llm_calls += 1
        self.patch_proposal_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
