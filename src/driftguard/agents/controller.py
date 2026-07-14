from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from driftguard.contracts.input_validator import InputValidationError, InputValidator
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.evidence.models import EvaluatorView
from driftguard.experiments.budgets import BudgetExhausted, BudgetTracker
from driftguard.llm import (
    ErrorCategory, InvalidStructuredOutput, LLMCache, LLMProvider, ModelConfig,
    ProviderError, ProviderRequest, ProviderTimeout, RateLimited, UsageCounter,
)
from driftguard.sandbox.evaluator import TaskEvaluator
from driftguard.sandbox.service import SandboxService

from .action_parser import ActionParser
from .context_builder import AgentContextBuilder
from .models import ActionType, AgentRunResult
from .task_memory import TaskMemory


class ToolAgentController:
    def __init__(
        self,
        provider: LLMProvider,
        model_config: ModelConfig,
        policy: Any,
        service: SandboxService,
        tracker: BudgetTracker,
        action_schema: dict[str, Any],
        base_prompt: str,
        policy_prompt: str = "",
        cache: LLMCache | None = None,
        force_refresh: bool = False,
    ):
        self.provider, self.model_config, self.policy, self.service = provider, model_config, policy, service
        self.tracker, self.action_schema = tracker, action_schema
        self.base_prompt, self.policy_prompt = base_prompt, policy_prompt
        self.cache, self.force_refresh = cache, force_refresh
        self.parser, self.context_builder = ActionParser(action_schema), AgentContextBuilder()
        self.validator, self.evaluator = InputValidator(), TaskEvaluator()
        self.memory, self.usage = TaskMemory(persistent=False), UsageCounter()

    def run(
        self,
        task: dict[str, Any],
        displayed_spec: dict[str, Any],
        public_scenario_id: str,
        episode: int,
        repetition: int,
    ) -> AgentRunResult:
        if isinstance(displayed_spec, EvaluatorView):
            raise TypeError("ToolAgentController cannot receive EvaluatorView")
        if not isinstance(displayed_spec, dict):
            raise TypeError("displayed_spec must be an Agent-visible document")
        assert_agent_visible({"task": task.get("instruction", ""), "displayed_spec": displayed_spec})
        initial_state = self.service.store.snapshot()
        actions: list[dict[str, Any]] = []
        trace_refs: list[str] = []
        last_evaluation = self._evaluate(task, initial_state, {}, [])
        termination, error, answer = "TASK_FAILED", None, None
        pending_exact_retry: dict[str, Any] | None = None
        try:
            while True:
                if pending_exact_retry is not None:
                    result, evaluation = self._call_tool(task, initial_state, pending_exact_retry["tool_id"], pending_exact_retry["arguments"])
                    actions.append({
                        "action_type": "EXACT_RETRY", **pending_exact_retry,
                        "executed_arguments": pending_exact_retry["arguments"], "result": result.to_dict(),
                    })
                    trace_refs.append(f"tool-{self.tracker.tool_calls:03d}")
                    last_evaluation = evaluation
                    pending_exact_retry = None
                    if evaluation["task_success"]:
                        termination = "TASK_SUCCESS"
                        break
                    decision = self.policy.after_failure(actions[-1], result.to_dict())
                    if decision.stop:
                        break
                    continue

                messages = self.context_builder.build(
                    self.base_prompt, self.policy_prompt, task["instruction"], displayed_spec,
                    self.memory.visible(), self._remaining(),
                )
                action = self._next_action(messages, public_scenario_id, episode, repetition)
                action_dict = action.to_dict()
                actions.append(action_dict)
                if action.action_type == ActionType.ABSTAIN:
                    termination = "ABSTAINED"
                    break
                if action.action_type == ActionType.FINAL_ANSWER:
                    answer = action.answer
                    last_evaluation = self._evaluate(task, initial_state, {"final_answer": answer}, [])
                    termination = "TASK_SUCCESS" if last_evaluation["task_success"] else "TASK_FAILED"
                    break
                if action.action_type == ActionType.REQUEST_PROBE:
                    if not self.policy.allows_probe:
                        termination, error = "SAFETY_BLOCKED", ErrorCategory.SAFETY_BLOCKED.value
                        break
                    self.tracker.consume_tool(probe=True)
                    probe = self.service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin")
                    self.memory.append({"probe_type": action.probe_type, "success": probe.ok, "target_tool_id": action.target_tool_id})
                    trace_refs.append(f"probe-{self.tracker.probe_calls:03d}")
                    continue
                if action.tool_id not in self.service.registry.operation_ids():
                    termination, error = "UNKNOWN_TOOL", ErrorCategory.UNKNOWN_TOOL.value
                    break
                contract = self.service.registry.require(action.tool_id)
                try:
                    normalized = self.validator.validate(contract, action.arguments)
                except InputValidationError as exc:
                    self.memory.append({
                        "tool_id": action.tool_id, "arguments": action.arguments,
                        "local_validation": {"valid": False, "error": str(exc), "field": exc.field},
                    })
                    termination, error = "INVALID_ARGUMENTS", ErrorCategory.INVALID_ARGUMENTS.value
                    break
                result, evaluation = self._call_tool(task, initial_state, action.tool_id, normalized)
                actions[-1]["executed_arguments"] = normalized
                trace_refs.append(f"tool-{self.tracker.tool_calls:03d}")
                last_evaluation = evaluation
                visible = {"tool_id": action.tool_id, "arguments": normalized, "response": result.to_dict()}
                self.memory.append(visible)
                if evaluation["task_success"]:
                    termination = "TASK_SUCCESS"
                    break
                decision = self.policy.after_failure(
                    {**visible, "local_validation": {"valid": True}}, result.to_dict()
                )
                if decision.context:
                    self.memory.append(decision.context)
                if decision.exact_retry:
                    pending_exact_retry = {"tool_id": action.tool_id, "arguments": normalized}
                if decision.stop:
                    break
        except BudgetExhausted:
            termination, error = "BUDGET_EXHAUSTED", ErrorCategory.BUDGET_EXHAUSTED.value
        except ProviderTimeout:
            termination, error = "PROVIDER_TIMEOUT", ErrorCategory.PROVIDER_TIMEOUT.value
        except RateLimited:
            termination, error = "RATE_LIMITED", ErrorCategory.RATE_LIMITED.value
        except ProviderError:
            termination, error = "PROVIDER_ERROR", ErrorCategory.PROVIDER_ERROR.value
        except InvalidStructuredOutput:
            termination, error = "INVALID_STRUCTURED_OUTPUT", ErrorCategory.INVALID_STRUCTURED_OUTPUT.value
        finally:
            self.memory.end_task()
            self.policy.end_task()
        return AgentRunResult(
            bool(last_evaluation["task_success"]), termination, error, answer,
            self.tracker.tool_calls, self.tracker.probe_calls, self.tracker.llm_calls,
            self.tracker.input_tokens, self.tracker.output_tokens, self.usage.latency_ms,
            tuple(trace_refs), tuple(actions), last_evaluation,
        )

    def _next_action(self, messages, public_scenario_id, episode, repetition):
        schema_hash = hashlib.sha256(json.dumps(self.action_schema, sort_keys=True).encode()).hexdigest()
        for repair_index in range(self.tracker.budget.max_format_repairs + 1):
            request = ProviderRequest(
                tuple(messages), self.action_schema,
                hashlib.sha256("\n".join(message["content"] for message in messages).encode()).hexdigest(),
                public_scenario_id, episode, self.policy.method, repetition,
            )
            key = LLMCache.key(self.model_config, request, schema_hash) if self.cache else None
            response = None if self.force_refresh or self.cache is None else self.cache.get(key)
            if response is None:
                self.tracker.ensure_llm_call_allowed()
                response = self.provider.complete(request)
                if self.cache is not None and key is not None:
                    self.cache.put(key, response)
            else:
                self.tracker.ensure_llm_call_allowed()
            self.tracker.consume_llm(response.input_tokens, response.output_tokens)
            self.usage.add(response)
            try:
                return self.parser.parse(response.raw_text)
            except (InvalidStructuredOutput, ValueError):
                if repair_index >= self.tracker.budget.max_format_repairs:
                    raise InvalidStructuredOutput("format repair budget exhausted")
                self.tracker.consume_format_repair()
                messages = tuple(messages) + ({
                    "role": "user", "content": "Return only a JSON object that validates against the supplied action schema.",
                },)
        raise AssertionError("unreachable")

    def _call_tool(self, task, initial_state, tool_id, arguments):
        self.tracker.consume_tool()
        result = self.service.call_tool(tool_id, arguments, task.get("actor_id", "agent_admin"))
        failures = [] if result.ok else [f"tool failed: {result.payload.get('error', {})}"]
        evaluation = self._evaluate(task, initial_state, {"tool_succeeded": result.ok}, failures)
        return result, evaluation

    def _evaluate(self, task, initial_state, answer, failures):
        return self.evaluator.evaluate(
            task, initial_state, self.service.store.snapshot(), answer,
            self.tracker.tool_calls, self.tracker.tool_calls, failures,
        ).to_dict()

    def _remaining(self) -> dict[str, int]:
        return {
            "llm_calls": self.tracker.budget.max_llm_calls - self.tracker.llm_calls,
            "tool_calls": self.tracker.budget.max_tool_calls - self.tracker.tool_calls,
            "probes": self.tracker.budget.max_probe_calls - self.tracker.probe_calls,
            "total_interactions": self.tracker.budget.max_total_interactions - self.tracker.total_interactions,
        }
