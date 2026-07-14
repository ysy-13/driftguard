from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

from driftguard.llm.errors import ErrorCategory


class BudgetExhausted(RuntimeError):
    category = ErrorCategory.BUDGET_EXHAUSTED


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
    _started: float = field(default_factory=time.monotonic)

    @property
    def total_interactions(self) -> int:
        return self.llm_calls + self.tool_calls

    def _check_values(self, llm_calls: int, tool_calls: int, probe_calls: int, input_tokens: int, output_tokens: int, format_repairs: int) -> None:
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
