from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from driftguard.contracts.input_validator import InputValidationError, InputValidator
from driftguard.contracts.registry import ContractRegistry
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.evidence import EvidenceCollector, EvidenceStore
from driftguard.evidence.models import EvaluatorView
from driftguard.experiments.budgets import BudgetExhausted, BudgetTracker
from driftguard.injection import AgentErrorHook
from driftguard.llm import (
    ErrorCategory, InvalidStructuredOutput, LLMCache, LLMProvider, ModelConfig,
    ProviderError, ProviderRequest, ProviderTimeout, RateLimited, UsageCounter,
)
from driftguard.sandbox.evaluator import TaskEvaluator
from driftguard.sandbox.service import SandboxService

from .action_parser import ActionParser
from .capabilities import PolicyCapabilities
from .context_builder import AgentContextBuilder
from .models import ActionType, AgentRunResult
from .state_machine import ControllerStateMachine
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
        config_hash: str = "",
        run_mode: str = "end_to_end",
        replay_normalized_runtime_arguments: bool = False,
    ):
        self.provider, self.model_config, self.policy, self.service = provider, model_config, policy, service
        self.tracker, self.action_schema = tracker, action_schema
        self.base_prompt, self.policy_prompt = base_prompt, policy_prompt
        self.cache, self.force_refresh = cache, force_refresh
        self.config_hash, self.run_mode = config_hash, run_mode
        self.replay_normalized_runtime_arguments = replay_normalized_runtime_arguments
        self.parser = ActionParser(action_schema)
        self.feedback_loop_enabled = "TOOL_CATALOG_RENDERER_V1" in base_prompt
        self.context_builder = AgentContextBuilder(full_catalog=self.feedback_loop_enabled)
        self.validator, self.evaluator = InputValidator(), TaskEvaluator()
        self.memory, self.usage = TaskMemory(persistent=False), UsageCounter()
        self.last_evidence_trace = None
        self.capabilities = PolicyCapabilities.for_method(self.policy.method)
        if self.capabilities.allows_probe != bool(self.policy.allows_probe):
            raise ValueError("PolicyCapabilities and policy safety gate disagree")

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
        evidence_store = EvidenceStore(f"trace-{public_scenario_id}", public_scenario_id)
        evidence_collector = EvidenceCollector(evidence_store, displayed_spec)
        displayed_registry = ContractRegistry(displayed_spec) if self.feedback_loop_enabled else self.service.registry
        actions: list[dict[str, Any]] = []
        trace_refs: list[str] = []
        policy_events: list[dict[str, Any]] = []
        agent_faults = AgentErrorHook()
        agent_call_fault_consumed = False
        agent_interpretation_fault_consumed = False
        machine = ControllerStateMachine()
        machine.move("TASK_RECEIVED", "TASK_ACCEPTED")
        last_evaluation = self._evaluate(task, initial_state, {}, [])
        termination, error, answer = "TASK_FAILED", None, None
        pending_exact_retry: dict[str, Any] | None = None
        try:
            while True:
                if pending_exact_retry is not None:
                    machine.move("TOOL_EXECUTION", "TRANSIENT_EXACT_RETRY")
                    result, evaluation = self._call_tool(task, initial_state, pending_exact_retry["tool_id"], pending_exact_retry["arguments"])
                    actions.append({
                        "action_type": "EXACT_RETRY", **pending_exact_retry,
                        "executed_arguments": pending_exact_retry["arguments"], "result": result.to_dict(),
                    })
                    trace_refs.append(f"tool-{self.tracker.tool_calls:03d}")
                    last_evaluation = evaluation
                    pending_exact_retry = None
                    machine.move("TASK_EVALUATION", "TOOL_RESULT_EVALUATED")
                    if evaluation["task_success"]:
                        termination = "TASK_SUCCESS"
                        machine.move("SUCCESS", "BUSINESS_GOAL_SATISFIED")
                        break
                    decision = self.policy.after_failure(actions[-1], result.to_dict())
                    policy_events.append(_policy_event(self.policy.method, "runtime_failure", decision))
                    machine.move("POLICY_RECOVERY", "RUNTIME_FAILURE_POLICY_HOOK")
                    if decision.context:
                        self.memory.append(decision.context)
                    if decision.stop:
                        break
                    machine.move("NEXT_MODEL_ACTION", "POLICY_REQUESTED_MODEL_ACTION")
                    continue

                machine.move("MODEL_ACTION", "INITIAL_MODEL_ACTION" if not actions else "RECOVERY_MODEL_ACTION")
                messages = self.context_builder.build(
                    self.base_prompt, self.policy_prompt, task["instruction"], displayed_spec,
                    self.memory.visible(), self._remaining(),
                    {"method": self.capabilities.method, "allowed_actions": list(self.capabilities.allowed_actions)},
                )
                action = self._next_action(messages, public_scenario_id, episode, repetition)
                action_dict = action.to_dict()
                actions.append(action_dict)
                raw_model_action = deepcopy(self.parser.last_raw_action or action_dict)
                actions[-1]["action_normalization_audit"] = {
                    "raw_model_action": raw_model_action,
                    "normalized_action": deepcopy(raw_model_action),
                    "normalization_diff": [],
                    "normalization_reason_code": "JSON_OBJECT_SCHEMA_VALIDATED_NO_SEMANTIC_CHANGE",
                    "semantic_change": False,
                }
                machine.move("ACTION_VALIDATION", f"ACTION_{action.action_type.value}_PARSED")
                if action.action_type == ActionType.ABSTAIN:
                    termination = "ABSTAINED"
                    machine.move("ABSTAINED", "AGENT_ABSTAINED")
                    break
                if action.action_type == ActionType.FINAL_ANSWER:
                    answer = action.answer
                    last_evaluation = self._evaluate(task, initial_state, {"final_answer": answer}, [])
                    machine.move("TASK_EVALUATION", "FINAL_ANSWER_EVALUATED")
                    if last_evaluation["task_success"]:
                        termination = "TASK_SUCCESS"
                        machine.move("SUCCESS", "BUSINESS_GOAL_AND_ANSWER_SATISFIED")
                        break
                    if not self.feedback_loop_enabled:
                        termination = "TASK_FAILED"
                        break
                    self.memory.append({
                        "evaluation_feedback": {
                            "category": "TASK_INCOMPLETE",
                            "message": "The final answer does not yet satisfy the visible task conditions.",
                            "remaining_budget": self._remaining(),
                        }
                    })
                    machine.move("POLICY_RECOVERY", "FINAL_ANSWER_DID_NOT_SATISFY_TASK")
                    machine.move("NEXT_MODEL_ACTION", "TASK_REMAINS_ACTIVE")
                    continue
                if action.action_type == ActionType.REQUEST_PROBE:
                    if not self.policy.allows_probe:
                        termination, error = "SAFETY_BLOCKED", ErrorCategory.SAFETY_BLOCKED.value
                        machine.move("SAFETY_BLOCKED", "ACTION_NOT_ALLOWED_BY_POLICY")
                        break
                    self.tracker.consume_tool(probe=True)
                    repo_id = task.get("bindings", {}).get("repo_id")
                    if repo_id is None:
                        termination, error = "SAFETY_BLOCKED", ErrorCategory.SAFETY_BLOCKED.value
                        machine.move("SAFETY_BLOCKED", "PROBE_TARGET_NOT_PUBLICLY_BOUND")
                        break
                    machine.move("TOOL_EXECUTION", "AUTHORIZED_READ_ONLY_PROBE")
                    probe = self.service.call_tool("get_repository", {"repo_id": repo_id}, task.get("actor_id", "agent_admin"))
                    self.memory.append({"probe_type": action.probe_type, "success": probe.ok, "target_tool_id": action.target_tool_id})
                    trace_refs.append(f"probe-{self.tracker.probe_calls:03d}")
                    machine.move("POLICY_RECOVERY", "PROBE_OBSERVATION_RECORDED")
                    machine.move("NEXT_MODEL_ACTION", "PROBE_COMPLETED")
                    continue
                if action.tool_id not in self.service.registry.operation_ids():
                    termination, error = "UNKNOWN_TOOL", ErrorCategory.UNKNOWN_TOOL.value
                    machine.move("SAFETY_BLOCKED", "UNKNOWN_TOOL_REJECTED")
                    break
                contract = displayed_registry.require(action.tool_id)
                proposed_arguments = action.arguments
                context = self.service.execution_context
                if (
                    self.feedback_loop_enabled and context is not None
                    and not agent_call_fault_consumed
                    and action.tool_id == context.profile.target_tool
                    and agent_faults.active(context)
                ):
                    proposed_arguments = agent_faults.inject_call(context, proposed_arguments)
                    agent_call_fault_consumed = True
                    actions[-1]["agent_fault_applied"] = True
                    if proposed_arguments != action.arguments:
                        actions[-1]["agent_fault_arguments"] = deepcopy(proposed_arguments)
                try:
                    normalized = self.validator.validate(contract, proposed_arguments)
                except InputValidationError as exc:
                    if not self.feedback_loop_enabled:
                        self.memory.append({
                            "tool_id": action.tool_id, "arguments": action.arguments,
                            "local_validation": {"valid": False, "error": str(exc), "field": exc.field},
                        })
                        termination, error = "INVALID_ARGUMENTS", ErrorCategory.INVALID_ARGUMENTS.value
                        break
                    feedback = self._validation_feedback(action.tool_id, proposed_arguments, contract, exc)
                    actions[-1]["local_validation"] = feedback["validation_error"]
                    self.memory.append(feedback)
                    machine.move("VALIDATION_FEEDBACK", "LOCAL_INPUT_VALIDATION_FAILED")
                    decision = self.policy.after_failure(
                        {"tool_id": action.tool_id, "arguments": proposed_arguments, "local_validation": {"valid": False, **feedback["validation_error"]}},
                        {"success": False, "error": feedback["validation_error"]},
                    )
                    policy_events.append(_policy_event(self.policy.method, "local_validation", decision))
                    machine.move("POLICY_RECOVERY", f"{self.policy.method.upper()}_VALIDATION_HOOK")
                    if decision.context:
                        self.memory.append(decision.context)
                    if self.policy.method.startswith("driftguard_"):
                        event = evidence_collector.add(
                            "local_validation", episode, action.tool_id,
                            displayed_request={"arguments": proposed_arguments},
                            local_validation_result=feedback["validation_error"],
                            normalized_observation={"kind": "validation_failure"},
                        )
                        self.memory.append({
                            "evidence": {
                                "event_ref": event.event_id, "kind": "validation_failure",
                                **feedback["validation_error"],
                            }
                        })
                    if decision.stop:
                        termination, error = "INVALID_ARGUMENTS", ErrorCategory.INVALID_ARGUMENTS.value
                        break
                    machine.move("NEXT_MODEL_ACTION", "VALIDATION_FEEDBACK_AVAILABLE")
                    continue
                defaulted_fields = sorted(set(normalized) - set(proposed_arguments))
                actions[-1]["displayed_input_audit"] = {
                    "proposed_arguments": deepcopy(proposed_arguments),
                    "validated_arguments": deepcopy(normalized),
                    "diff": [
                        {
                            "operation": "add",
                            "field": field,
                            "value": deepcopy(normalized[field]),
                            "source": "displayed_spec_property_default",
                        }
                        for field in defaulted_fields
                    ],
                    "reason_code": (
                        "DISPLAYED_SCHEMA_DEFAULT_APPLIED"
                        if defaulted_fields else "NO_DEFAULT_APPLIED"
                    ),
                    "defaulted_fields": defaulted_fields,
                }
                machine.move("TOOL_EXECUTION", "LOCAL_INPUT_VALIDATION_PASSED")
                runtime_arguments = (
                    normalized
                    if context is None or self.replay_normalized_runtime_arguments
                    else proposed_arguments
                )
                result, evaluation = self._call_tool(task, initial_state, action.tool_id, runtime_arguments)
                actions[-1]["executed_arguments"] = runtime_arguments
                trace_refs.append(f"tool-{self.tracker.tool_calls:03d}")
                last_evaluation = evaluation
                visible = {"tool_id": action.tool_id, "arguments": runtime_arguments, "response": result.to_dict()}
                if (
                    self.feedback_loop_enabled and context is not None
                    and not agent_interpretation_fault_consumed
                    and action.tool_id == context.profile.target_tool
                ):
                    interpretation = agent_faults.interpret(context, result)
                    if interpretation is not None:
                        agent_interpretation_fault_consumed = True
                        visible["response_interpretation"] = interpretation
                self.memory.append(visible)
                machine.move("TASK_EVALUATION", "TOOL_RESULT_EVALUATED")
                if evaluation["task_success"]:
                    termination = "TASK_SUCCESS"
                    machine.move("SUCCESS", "BUSINESS_GOAL_SATISFIED")
                    break
                decision = self.policy.after_failure(
                    {**visible, "local_validation": {"valid": True}}, result.to_dict()
                )
                policy_events.append(_policy_event(self.policy.method, "runtime_or_task_incomplete", decision))
                if decision.context:
                    self.memory.append(decision.context)
                machine.move("POLICY_RECOVERY", "TASK_INCOMPLETE_OR_RUNTIME_FAILURE")
                if decision.exact_retry:
                    pending_exact_retry = {"tool_id": action.tool_id, "arguments": runtime_arguments}
                if decision.stop:
                    break
                machine.move("NEXT_MODEL_ACTION", "POLICY_REQUESTED_RECOVERY")
        except BudgetExhausted:
            termination, error = "BUDGET_EXHAUSTED", ErrorCategory.BUDGET_EXHAUSTED.value
            if machine.state not in {"SUCCESS", "ABSTAINED", "SAFETY_BLOCKED"}:
                machine.move("BUDGET_EXHAUSTED", "INTERACTION_BUDGET_EXHAUSTED")
        except ProviderTimeout:
            termination, error = "PROVIDER_TIMEOUT", ErrorCategory.PROVIDER_TIMEOUT.value
            machine.move("PROVIDER_FAILURE", "PROVIDER_TIMEOUT")
        except RateLimited:
            termination, error = "RATE_LIMITED", ErrorCategory.RATE_LIMITED.value
            machine.move("PROVIDER_FAILURE", "PROVIDER_RATE_LIMITED")
        except ProviderError:
            termination, error = "PROVIDER_ERROR", ErrorCategory.PROVIDER_ERROR.value
            machine.move("PROVIDER_FAILURE", "PROVIDER_ERROR")
        except InvalidStructuredOutput:
            termination, error = "INVALID_STRUCTURED_OUTPUT", ErrorCategory.INVALID_STRUCTURED_OUTPUT.value
            machine.move("INVALID_STRUCTURED_OUTPUT", "FORMAT_REPAIR_EXHAUSTED")
        finally:
            self.last_evidence_trace = evidence_store.trace
            self.memory.end_task()
            self.policy.end_task()
        return AgentRunResult(
            bool(last_evaluation["task_success"]), termination, error, answer,
            self.tracker.tool_calls, self.tracker.probe_calls, self.tracker.llm_calls,
            self.tracker.input_tokens, self.tracker.output_tokens, self.usage.latency_ms,
            tuple(trace_refs), tuple(actions), last_evaluation, machine.visible(),
            self.context_builder.last_catalog_fingerprint, tuple(policy_events),
        )

    def _validation_feedback(self, tool_id, arguments, contract, exc: InputValidationError) -> dict[str, Any]:
        method = self.policy.method
        validation_error: dict[str, Any] = {
            "category": ErrorCategory.INVALID_ARGUMENTS.value,
            "target_tool": tool_id,
            "invalid_field_path": exc.field,
            "error_code": "LOCAL_SCHEMA_VALIDATION_ERROR",
            "message": "The proposed arguments do not satisfy the displayed tool contract.",
            "remaining_budget": self._remaining(),
        }
        if method in {"validation_guided", "driftguard_llm", "driftguard_symbolic"}:
            validation_error["message"] = str(exc)
        if method == "validation_guided":
            validation_error["displayed_schema_fragment"] = _schema_fragment(contract.input_schema, exc.field)
        return {
            "tool_id": tool_id,
            "arguments": arguments,
            "validation_error": validation_error,
        }

    def _next_action(self, messages, public_scenario_id, episode, repetition):
        schema_hash = hashlib.sha256(json.dumps(self.action_schema, sort_keys=True).encode()).hexdigest()
        for repair_index in range(self.tracker.budget.max_format_repairs + 1):
            request = ProviderRequest(
                tuple(messages), self.action_schema,
                hashlib.sha256("\n".join(message["content"] for message in messages).encode()).hexdigest(),
                public_scenario_id, episode, self.policy.method, repetition,
                self.model_config.seed, self.run_mode, self.config_hash,
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
            except (InvalidStructuredOutput, ValueError) as exc:
                if repair_index >= self.tracker.budget.max_format_repairs:
                    raise InvalidStructuredOutput("format repair budget exhausted")
                self.tracker.consume_format_repair()
                messages = tuple(messages) + ({
                    "role": "user", "content": _action_format_repair(
                        response.raw_text, exc, self.action_schema,
                    ),
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


def _action_format_repair(raw_text: str, error: Exception, schema: dict[str, Any]) -> str:
    issues = _action_field_issues(raw_text, schema)
    schema_text = json.dumps(schema, indent=2, sort_keys=True)
    return (
        f"Schema validation error: {error}\n"
        f"Missing or illegal fields: {issues}\n"
        "Correct the field names and action-specific field set. Return only one JSON object "
        "with no markdown or commentary. The complete AgentAction output schema is:\n"
        f"<OUTPUT_SCHEMA>\n{schema_text}\n</OUTPUT_SCHEMA>"
    )


def _action_field_issues(raw_text: str, schema: dict[str, Any]) -> str:
    try:
        value = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return "the response is not a valid JSON object"
    if not isinstance(value, dict):
        return "the JSON root must be an object"

    branches = [item for item in schema.get("oneOf", ()) if isinstance(item, dict)]
    if branches:
        allowed = {
            field
            for branch in branches
            for field in branch.get("properties", {})
        }
        required = {"action_type", "concise_decision_summary"}
        action_type = value.get("action_type")
        for branch in branches:
            expected = branch.get("properties", {}).get("action_type", {}).get("const")
            if expected == action_type:
                allowed = set(branch.get("properties", {}))
                required = set(branch.get("required", ()))
                break
    else:
        allowed = set(schema.get("properties", {}))
        required = set(schema.get("required", ()))

    missing = sorted(required - set(value))
    illegal = sorted(set(value) - allowed)
    parts = []
    if missing:
        parts.append("missing=" + ", ".join(missing))
    if illegal:
        parts.append("illegal=" + ", ".join(illegal))
    if not parts:
        parts.append("one or more field values violate their declared type or action-specific constraint")
    return "; ".join(parts)


def _schema_fragment(schema: dict[str, Any], field: str | None) -> dict[str, Any]:
    if not field:
        return {
            "type": schema.get("type"),
            "required": deepcopy(schema.get("required", [])),
            "additionalProperties": schema.get("additionalProperties"),
        }
    current: Any = schema
    for token in str(field).replace("[", ".").replace("]", "").split("."):
        if not token:
            continue
        properties = current.get("properties", {}) if isinstance(current, dict) else {}
        if token in properties:
            current = properties[token]
        elif isinstance(current, dict) and current.get("type") == "array":
            current = current.get("items", {})
        else:
            return {"field": field, "required": field in schema.get("required", [])}
    return {"field": field, "schema": deepcopy(current), "required": field in schema.get("required", [])}


def _policy_event(method: str, failure_kind: str, decision: Any) -> dict[str, Any]:
    context = decision.context or {}
    return {
        "method": method,
        "failure_kind": failure_kind,
        "exact_retry": bool(decision.exact_retry),
        "request_llm": bool(decision.request_llm),
        "stop": bool(decision.stop),
        "context_types": sorted(context),
    }
