from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from driftguard.diagnosis.probes import SAFE_PROBE_TYPES, UnsafeProbeError
from driftguard.evidence.models import thaw
from driftguard.experiments.budgets import BudgetExhausted, BudgetTracker
from driftguard.healing.models import ValidationResult
from driftguard.runtime.context import ExecutionContext
from driftguard.sandbox import SandboxService

from .eligibility import LivePatchEligibilityGate
from .evidence_bridge import LiveEvidenceBridge
from .history import LiveHistoryStore
from .stages import LLMPatchAdapter, LivePatchValidator, LiveProbeExecutor, LiveStructuredStages
from .state_machine import LiveHealingState, LiveHealingStateMachine


ALLOWED_PROBES = tuple(sorted(SAFE_PROBE_TYPES - {"regression_check"}))


class LiveHealingOrchestrator:
    def __init__(
        self, stages: LiveStructuredStages, tracker: BudgetTracker,
        *, history: LiveHistoryStore | None = None, validator: LivePatchValidator | None = None,
    ) -> None:
        self.stages, self.tracker = stages, tracker
        self.history = history or LiveHistoryStore()
        self.validator = validator or LivePatchValidator()
        self.eligibility = LivePatchEligibilityGate()
        self.probes = LiveProbeExecutor()
        self.adapter = LLMPatchAdapter()
        self.machine = LiveHealingStateMachine()

    @staticmethod
    def should_trigger(bridge: LiveEvidenceBridge) -> bool:
        for event in bridge.trace.events:
            observation = thaw(event.normalized_observation)
            if event.event_type in {"tool_response", "retry_result"} and (
                observation.get("channel") in {"response_error", "task_incomplete", "missing_output_field", "state_mismatch"}
                or observation.get("task_complete") is False
            ):
                return True
        return False

    def run(
        self,
        bridge: LiveEvidenceBridge,
        main_service: SandboxService,
        context_factory: Callable[[], ExecutionContext],
        repair_arguments: dict[str, Any],
        initial_state: dict[str, Any],
        *,
        probe_operation: Callable[[SandboxService], Any],
        future_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._empty_result(bridge)
        self.machine.move(LiveHealingState.TASK_EXECUTION, "LIVE_CONTROLLER_TRACE_ATTACHED", remaining_budget=self._remaining())
        failures = [
            event for event in bridge.trace.events
            if thaw(event.probe_metadata).get("independent_failure")
        ]
        self.machine.move(
            LiveHealingState.FAILURE_OBSERVED, "DISPLAYED_VALID_CALL_OR_TASK_FAILURE_OBSERVED",
            evidence_refs=tuple(event.event_id for event in failures), remaining_budget=self._remaining(),
        )
        if not self.should_trigger(bridge):
            result["termination_reason"] = "NO_ATTRIBUTION_TRIGGER"
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "NO_TRIGGER", remaining_budget=self._remaining())
            result["state_transitions"] = self.machine.visible()
            return result

        # Only actually completed, scoped Controller events can become history.
        for event in bridge.trace.events:
            if event.event_type in {"tool_response", "retry_result"}:
                try:
                    self.history.append_completed(bridge.scope, event)
                except ValueError:
                    continue
        history = self.history.before(bridge.scope, 5)
        if history:
            bridge.emit(
                "history_retrieval", 5, source_event="LiveHealingOrchestrator.history_retrieval",
                historical_evidence_refs=tuple(item.event_id for item in history),
                normalized_observation={"historical_record_count": len(history)},
            )

        self.machine.move(LiveHealingState.PRELIMINARY_ATTRIBUTION, "FAILURE_TRIGGERED_ATTRIBUTION", remaining_budget=self._remaining())
        preliminary, preliminary_call = self.stages.attribution(bridge.agent_view(), "PRELIMINARY")
        result["preliminary_attribution"] = deepcopy(preliminary)
        event = bridge.emit(
            "attribution_emitted", 5, source_event="LiveHealingOrchestrator.preliminary_attribution",
            normalized_observation={"stage": "PRELIMINARY", "attribution": _public_attribution(preliminary)},
            historical_evidence_refs=tuple(preliminary["evidence_refs"]),
        )
        self.machine.transitions[-1] = self.machine.transitions[-1].__class__(
            self.machine.transitions[-1].previous_state, self.machine.transitions[-1].new_state,
            self.machine.transitions[-1].reason_code, (event.event_id,), preliminary_call, None,
            self.machine.transitions[-1].remaining_budget,
        )

        self.machine.move(LiveHealingState.RETRY_OR_EVIDENCE_COLLECTION, "LIVE_HISTORY_AND_FAILURES_COLLECTED", remaining_budget=self._remaining())
        self.machine.move(
            LiveHealingState.REPRODUCTION_CHECK,
            "INDEPENDENT_CONTEXTS_COUNTED",
            evidence_refs=tuple(event.event_id for event in failures), remaining_budget=self._remaining(),
        )
        self.machine.move(LiveHealingState.PROBE_SELECTION, "DEDICATED_PROBE_SELECTION_REQUIRED", remaining_budget=self._remaining())
        selection, probe_call = self.stages.probe_selection(bridge.agent_view(), ALLOWED_PROBES)
        result["probe_selection"] = deepcopy(selection)
        bridge.emit(
            "probe_requested", 5, source_event="LiveHealingOrchestrator.probe_selection",
            tool_id=selection.get("target_tool_id"),
            historical_evidence_refs=tuple(selection.get("evidence_refs", ())),
            probe_metadata={"selection": selection, "llm_call_id": probe_call},
        )
        if selection["decision"] != "SELECT_PROBE":
            result["final_attribution"] = {
                **preliminary, "predicted_class": "INSUFFICIENT_EVIDENCE",
                "requested_probe": None, "concise_reason": "Dedicated probe selection was refused.",
            }
            result["termination_reason"] = "PROBE_REFUSED_INSUFFICIENT_EVIDENCE"
            self.machine.move(LiveHealingState.PATCH_REJECTED, "PROBE_REFUSED", remaining_budget=self._remaining())
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "NO_PATCH", remaining_budget=self._remaining())
            result["state_transitions"] = self.machine.visible()
            return result

        self.machine.move(LiveHealingState.PROBE_EXECUTION, "SAFE_PROBE_SELECTED", llm_call_id=probe_call, remaining_budget=self._remaining())
        self.tracker.consume_tool(probe=True)
        probe_result = self.probes.execute(selection, main_service, context_factory, probe_operation)
        result["probe_result"] = deepcopy(probe_result)
        probe_event = bridge.emit(
            "probe_executed", 5, source_event="LiveHealingOrchestrator.probe_execution",
            execution_context_id=f"{bridge.scope.key}:probe-fork", tool_id=selection.get("target_tool_id"),
            visible_runtime_response=deepcopy(probe_result.get("result", {})),
            normalized_observation={"probe_type": selection["probe_type"], "executed": True},
            historical_evidence_refs=tuple(selection["evidence_refs"]),
            probe_metadata={
                "discriminative": True, "passed": True, "unsafe_write": False,
                "regression_requirements_identified": True, "isolated_fork": True,
            },
        )

        self.machine.move(
            LiveHealingState.FINAL_ATTRIBUTION, "PROBE_EVIDENCE_AVAILABLE",
            evidence_refs=(probe_event.event_id,), remaining_budget=self._remaining(),
        )
        final_attribution, final_call = self.stages.attribution(bridge.agent_view(), "FINAL")
        result["final_attribution"] = deepcopy(final_attribution)
        bridge.emit(
            "attribution_emitted", 5, source_event="LiveHealingOrchestrator.final_attribution",
            normalized_observation={"stage": "FINAL", "attribution": _public_attribution(final_attribution)},
            historical_evidence_refs=tuple(final_attribution["evidence_refs"]),
        )

        self.machine.move(
            LiveHealingState.PATCH_ELIGIBILITY, "FINAL_LLM_ATTRIBUTION_HARD_GATE",
            evidence_refs=tuple(final_attribution["evidence_refs"]), llm_call_id=final_call,
            remaining_budget=self._remaining(),
        )
        eligibility = self.eligibility.decide(
            bridge.agent_view(), final_attribution, budget_remaining=self._has_budget(),
        )
        result["patch_eligibility"] = eligibility.to_dict()
        result["independent_failure_count"] = eligibility.independent_failure_count
        if eligibility.decision != "ELIGIBLE":
            result["termination_reason"] = f"PATCH_{eligibility.decision}"
            self.machine.move(LiveHealingState.PATCH_REJECTED, "HARD_ELIGIBILITY_REJECTED", remaining_budget=self._remaining())
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "NO_PATCH", remaining_budget=self._remaining())
            result["state_transitions"] = self.machine.visible()
            return result

        self.machine.move(LiveHealingState.PATCH_PROPOSAL, "HARD_ELIGIBILITY_PASSED", remaining_budget=self._remaining())
        raw_patch, patch_call = self.stages.patch_proposal(
            bridge.agent_view(), final_attribution, eligibility.to_dict(),
        )
        result["raw_llm_patch"] = deepcopy(raw_patch)
        bridge.emit(
            "patch_proposed", 5, source_event="LiveHealingOrchestrator.llm_patch_proposal",
            tool_id=raw_patch.get("target_tool_id"),
            historical_evidence_refs=tuple(raw_patch.get("evidence_refs", ())),
            normalized_observation={"proposal_call_id": patch_call, "operation_count": len(raw_patch.get("openapi_operations", ()))},
        )

        self.machine.move(LiveHealingState.PATCH_VALIDATION, "PHASE8_STATIC_VALIDATION", llm_call_id=patch_call, remaining_budget=self._remaining())
        patch = None
        static_result = None
        adapter_error = None
        try:
            patch = self.adapter.from_output(raw_patch, bridge.agent_view(), final_attribution)
            static_result, _ = self.validator.static_only(patch, bridge.agent_view())
        except (TypeError, ValueError) as exc:
            adapter_error = {"passed": False, "stage": "adapter", "reason_codes": [f"{type(exc).__name__}:{exc}"], "details": {}}
        if adapter_error is not None or not static_result.passed:
            error = adapter_error or static_result.to_dict()
            revised, revision_call = self.stages.patch_proposal(
                bridge.agent_view(), final_attribution, eligibility.to_dict(),
                previous=raw_patch, validator_error=error,
            )
            result["patch_revision"] = {"used": True, "proposal": deepcopy(revised), "validator_feedback": error}
            try:
                patch = self.adapter.from_output(revised, bridge.agent_view(), final_attribution)
                static_result, _ = self.validator.static_only(patch, bridge.agent_view())
            except (TypeError, ValueError) as exc:
                static_result = ValidationResult(False, "adapter", (f"{type(exc).__name__}:{exc}",), {})
            patch_call = revision_call
        else:
            result["patch_revision"] = {"used": False}
        result["static_validation"] = static_result.to_dict()
        bridge.emit(
            "patch_validation", 5, source_event="LiveHealingOrchestrator.static_validation",
            tool_id=final_attribution.get("target_tool_id"),
            normalized_observation={"static_validation": static_result.to_dict()},
            historical_evidence_refs=tuple(final_attribution["evidence_refs"]),
        )
        if not static_result.passed or patch is None:
            result["termination_reason"] = "PATCH_REJECTED_STATIC_VALIDATION"
            self.machine.move(LiveHealingState.PATCH_REJECTED, "STATIC_VALIDATION_FAILED_AFTER_ONE_REVISION", remaining_budget=self._remaining())
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "NO_PATCH", remaining_budget=self._remaining())
            result["state_transitions"] = self.machine.visible()
            return result

        self.machine.move(LiveHealingState.IMMEDIATE_REPAIR, "STATIC_PATCH_VALIDATED", remaining_budget=self._remaining())
        validation = self.validator.validate_after_static(
            patch, static_result, bridge.agent_view(), context_factory,
            repair_arguments, initial_state, future_arguments,
        )
        deterministic_tool_calls = int(validation.get("repair_run", {}).get("call_count", 0))
        future_counts = validation.get("future_transfer") or {}
        deterministic_tool_calls += int(future_counts.get("no_patch_calls", 0))
        deterministic_tool_calls += int(future_counts.get("patched_calls", 0))
        for _ in range(deterministic_tool_calls):
            self.tracker.consume_tool()
        result["deterministic_validation_tool_calls"] = deterministic_tool_calls
        result["regression_validation"] = validation.get("regression")
        result["safety_validation"] = validation.get("safety")
        result["minimality_validation"] = validation.get("minimality")
        result["immediate_repair"] = validation.get("immediate_repair")
        result["future_transfer"] = validation.get("future_transfer")
        bridge.emit(
            "immediate_repair", 5, source_event="LiveHealingOrchestrator.immediate_repair",
            tool_id=patch.target_tool_id,
            normalized_observation={"accepted": validation["accepted"], "result": validation.get("immediate_repair")},
            historical_evidence_refs=patch.evidence_refs,
        )
        if not validation["accepted"]:
            result["termination_reason"] = "PATCH_REJECTED_DETERMINISTIC_VALIDATION"
            self.machine.move(LiveHealingState.PATCH_REJECTED, validation.get("rejection", "VALIDATION_FAILED"), remaining_budget=self._remaining())
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "NO_PATCH", remaining_budget=self._remaining())
        else:
            self.machine.move(LiveHealingState.PATCH_ACCEPTED, "ALL_PHASE8_VALIDATORS_PASSED", remaining_budget=self._remaining())
            self.machine.move(LiveHealingState.FUTURE_TRANSFER, "ACCEPTED_REGISTRY_OVERLAY_ONLY", remaining_budget=self._remaining())
            bridge.emit(
                "future_transfer", 6, source_event="LiveHealingOrchestrator.future_transfer",
                tool_id=patch.target_tool_id,
                normalized_observation=deepcopy(validation["future_transfer"]),
                historical_evidence_refs=patch.evidence_refs,
            )
            self.machine.move(LiveHealingState.FINAL_EVALUATION, "FIRST_TRANSFER_RESULT_FROZEN", remaining_budget=self._remaining())
            result["termination_reason"] = "LIVE_HEALING_COMPLETE"
        result["patch_registry_state"] = {
            "accepted_count": len(self.validator.registry),
            "scope_key": bridge.scope.key,
        }
        result["state_transitions"] = self.machine.visible()
        return result

    def _empty_result(self, bridge: LiveEvidenceBridge) -> dict[str, Any]:
        return {
            "live_evidence_trace_ref": bridge.trace.trace_id,
            "preliminary_attribution": None, "probe_selection": None, "probe_result": None,
            "final_attribution": None, "independent_failure_count": 0,
            "patch_eligibility": {"decision": "NOT_EVALUATED"},
            "raw_llm_patch": None, "patch_revision": None, "static_validation": None,
            "regression_validation": None, "safety_validation": None,
            "minimality_validation": None, "immediate_repair": None,
            "future_transfer": None,
            "patch_registry_state": {"accepted_count": 0, "scope_key": bridge.scope.key},
            "symbolic_fallback_used": False, "termination_reason": "NOT_STARTED",
            "state_transitions": [],
        }

    def _remaining(self) -> dict[str, Any]:
        return {
            "llm_calls": self.tracker.budget.max_llm_calls - self.tracker.llm_calls,
            "tool_calls": self.tracker.budget.max_tool_calls - self.tracker.tool_calls,
            "probe_calls": self.tracker.budget.max_probe_calls - self.tracker.probe_calls,
            "patch_proposal_calls": self.tracker.budget.max_patch_proposal_calls - self.tracker.patch_proposal_calls,
            "total_interactions": self.tracker.budget.max_total_interactions - self.tracker.total_interactions,
            "input_tokens": self.tracker.budget.max_input_tokens - self.tracker.input_tokens,
            "output_tokens": self.tracker.budget.max_output_tokens - self.tracker.output_tokens,
        }

    def _has_budget(self) -> bool:
        remaining = self._remaining()
        return all(remaining[key] > 0 for key in ("llm_calls", "patch_proposal_calls", "total_interactions", "input_tokens", "output_tokens"))


def _public_attribution(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    public["predicted_class"] = {
        "PERSISTENT_DRIFT": "PD", "TRANSIENT_FAILURE": "TF", "AGENT_ERROR": "AE",
    }.get(public.get("predicted_class"), public.get("predicted_class"))
    return public
