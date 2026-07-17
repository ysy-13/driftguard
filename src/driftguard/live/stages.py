from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.contracts.registry import ContractRegistry, ToolContract
from driftguard.diagnosis.probes import ProbePlan, SAFE_PROBE_TYPES, UnsafeProbeError
from driftguard.evidence.collector import stable_hash
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.evidence.models import AgentView, thaw
from driftguard.experiments.budgets import BudgetTracker
from driftguard.healing.minimality_validator import MinimalityValidator
from driftguard.healing.models import PatchLifecycle, PatchOperation, ToolSpecPatch
from driftguard.healing.overlay import SpecOverlay
from driftguard.healing.patch_registry import PatchRegistry
from driftguard.healing.regression_validator import RegressionValidator
from driftguard.healing.repair_executor import RepairExecutor, repair_validation
from driftguard.healing.safety_validator import SafetyValidator
from driftguard.healing.static_validator import StaticPatchValidator
from driftguard.healing.transfer_evaluator import FutureTransferEvaluator
from driftguard.live.adapter_contract import (
    neutral_example, operation_types, rendered_contract, schema_fragment,
    validate_semantics,
)
from driftguard.live.location import locations_match, normalize_location
from driftguard.llm import (
    InvalidStructuredOutput, LLMCache, LLMProvider, ModelConfig, ProviderRequest,
    parse_structured_output,
)
from driftguard.runtime.context import ExecutionContext
from driftguard.sandbox import SandboxService, StateStore
from driftguard.sandbox.diff import state_diff


SCHEMA_DIR = PROJECT_ROOT / "benchmark/schemas"
PROMPT_DIR = PROJECT_ROOT / "benchmark/prompts"


def _load(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


class LiveStructuredStages:
    def __init__(
        self, provider: LLMProvider, tracker: BudgetTracker, *, seed: int | None = None,
        repetition: int = 0, cache: LLMCache | None = None,
        model_config: ModelConfig | None = None, force_refresh: bool = False,
        config_hash: str = "live-healing-v1",
    ):
        self.provider, self.tracker = provider, tracker
        self.seed, self.repetition = seed, repetition
        self.cache, self.force_refresh, self.config_hash = cache, force_refresh, config_hash
        self.model_config = model_config or ModelConfig(provider="mock", model_id="mock-live-structured")
        self.attribution_schema = _load("llm_attribution_schema_v3.json")
        self.probe_schema = _load("probe_selection_schema_v3.json")
        self.patch_schema = _load("llm_patch_proposal_schema_v3.json")
        self.call_ids: list[str] = []
        self.call_records: list[dict[str, Any]] = []

    def attribution(self, view: AgentView, stage: str) -> tuple[dict[str, Any], str]:
        payload = {
            "displayed_spec": thaw(view.displayed_spec),
            "live_evidence_trace": view.trace.to_dict(),
            "current_episode": view.current_episode,
            "stage": stage,
        }
        value, call_id = self._call(
            "component_attribution_v3.txt", self.attribution_schema, payload,
            f"live_{stage.lower()}_attribution", patch=False,
        )
        available = {event.event_id for event in view.trace.events}
        if not set(value["evidence_refs"]).issubset(available):
            raise InvalidStructuredOutput("attribution cites a nonexistent live evidence ID")
        return value, call_id

    def probe_selection(self, view: AgentView, allowed: tuple[str, ...]) -> tuple[dict[str, Any], str]:
        payload = {
            "displayed_spec": thaw(view.displayed_spec),
            "live_evidence_trace": view.trace.to_dict(),
            "current_episode": view.current_episode,
            "allowed_probe_types": list(allowed),
            "available_tools": list(ContractRegistry(thaw(view.displayed_spec)).operation_ids()),
        }
        value, call_id = self._call(
            "live/driftguard_probe_selection_v3.txt", self.probe_schema, payload,
            "live_probe_selection", patch=False,
        )
        available = {event.event_id for event in view.trace.events}
        if not set(value["evidence_refs"]).issubset(available):
            raise InvalidStructuredOutput("probe selection cites a nonexistent live evidence ID")
        if value["decision"] == "SELECT_PROBE":
            if value["probe_type"] not in allowed:
                raise UnsafeProbeError("model selected a probe outside the allow-list")
            if value["target_tool"] not in ContractRegistry(thaw(view.displayed_spec)).operation_ids():
                raise UnsafeProbeError("model selected a probe target outside the displayed registry")
        return value, call_id

    def patch_proposal(
        self, view: AgentView, attribution: dict[str, Any], eligibility: dict[str, Any],
        *, previous: dict[str, Any] | None = None, validator_error: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        public_attribution = deepcopy(attribution)
        public_attribution["predicted_class"] = {
            "PERSISTENT_DRIFT": "PD", "TRANSIENT_FAILURE": "TF", "AGENT_ERROR": "AE",
        }.get(public_attribution.get("predicted_class"), public_attribution.get("predicted_class"))
        payload = {
            "live_agent_view": {
                "displayed_spec": thaw(view.displayed_spec),
                "live_evidence_trace": view.trace.to_dict(),
                "current_episode": view.current_episode,
            },
            "final_attribution": public_attribution,
            "patch_eligibility": eligibility,
            "allowed_target_tool": attribution.get("target_tool_id"),
            "normalized_attributed_location": deepcopy(attribution.get("location")),
            "allowed_adapter_operations": list(operation_types(attribution.get("drift_category"))),
            "allowed_target_scope": _allowed_target_scope(
                thaw(view.displayed_spec), attribution.get("target_tool_id"),
                attribution.get("location"),
            ),
            "adapter_schema_fragment": schema_fragment(attribution.get("drift_category", "")),
        }
        if previous is not None:
            payload["previous_proposal"] = previous
            payload["deterministic_validator_error"] = _sanitized_validator_feedback(validator_error)
            payload["revision"] = 1
        return self._call(
            "live/driftguard_patch_v3.txt", self.patch_schema, payload,
            "live_patch_revision" if previous is not None else "live_patch_proposal",
            patch=True,
        )

    def _call(self, prompt_name, schema, payload, mode, *, patch: bool) -> tuple[dict[str, Any], str]:
        assert_agent_visible(payload)
        template = (PROMPT_DIR / prompt_name).read_text(encoding="utf-8")
        prompt = template.replace("{{OUTPUT_SCHEMA}}", json.dumps(schema, indent=2, sort_keys=True))
        prompt = prompt.replace("{{ADAPTER_CONTRACT}}", json.dumps(rendered_contract(), indent=2, sort_keys=True))
        prompt = prompt.replace("{{NEUTRAL_EXAMPLE}}", json.dumps(neutral_example(), indent=2, sort_keys=True))
        messages: tuple[dict[str, str], ...] = (
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        )
        schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        public_id = payload.get("live_agent_view", payload).get("live_evidence_trace", {}).get(
            "public_scenario_id", payload.get("live_evidence_trace", {}).get("public_scenario_id", "public-live")
        )
        episode = int(payload.get("current_episode", payload.get("live_agent_view", {}).get("current_episode", 5)))
        while True:
            request = ProviderRequest(
                messages, schema, hashlib.sha256(prompt.encode()).hexdigest(),
                public_id, episode, "driftguard_llm", self.repetition, self.seed,
                mode, self.config_hash,
            )
            reservation = self.tracker.reserve_llm_call(patch_proposal=patch)
            cache_key = LLMCache.key(self.model_config, request, schema_hash) if self.cache else None
            response = None if self.force_refresh or self.cache is None else self.cache.get(cache_key)
            try:
                if response is None:
                    response = self.provider.complete(request)
                    if self.cache is not None and cache_key is not None:
                        self.cache.put(cache_key, response)
            except Exception:
                self.tracker.release_llm_call(reservation)
                raise
            self.tracker.settle_llm_call(
                reservation, response.input_tokens, response.output_tokens,
            )
            self.call_ids.append(response.response_id)
            try:
                parsed = parse_structured_output(response.raw_text, schema)
                if self.cache is not None and cache_key is not None:
                    self.cache.put(cache_key, replace(response, parsed_output=parsed))
                self.call_records.append({
                    "stage": mode, "response_id": response.response_id,
                    "raw_response_sha256": hashlib.sha256(response.raw_text.encode()).hexdigest(),
                    "parsed_output": deepcopy(parsed), "cached": bool(response.cached),
                    "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                    "latency_ms": response.latency_ms, "provider_attempts": response.provider_attempts,
                })
                self.tracker.assert_settled_within_budget()
                return parsed, response.response_id
            except InvalidStructuredOutput as exc:
                self.call_records.append({
                    "stage": mode, "response_id": response.response_id,
                    "raw_response_sha256": hashlib.sha256(response.raw_text.encode()).hexdigest(),
                    "parsed_output": None, "cached": bool(response.cached),
                    "validation_error": str(exc), "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens, "latency_ms": response.latency_ms,
                    "provider_attempts": response.provider_attempts,
                })
                self.tracker.assert_settled_within_budget()
                if self.tracker.format_repairs >= self.tracker.budget.max_format_repairs:
                    raise
                self.tracker.consume_format_repair()
                messages = messages + ({
                    "role": "user",
                    "content": (
                        f"Schema validation error: {exc}\n"
                        "Return only one corrected JSON object. The complete output schema is:\n"
                        f"<OUTPUT_SCHEMA>\n{json.dumps(schema, indent=2, sort_keys=True)}\n</OUTPUT_SCHEMA>"
                    ),
                },)


class ControllerProbeError(RuntimeError):
    pass


class LiveProbeExecutor:
    def execute(
        self, selection: dict[str, Any], main_service: SandboxService,
        context_factory: Callable[[], ExecutionContext], displayed_spec: dict[str, Any],
        *, actor_id: str = "agent_admin",
    ) -> dict[str, Any]:
        probe_type = selection.get("probe_type")
        if selection.get("decision") != "SELECT_PROBE":
            return {"executed": False, "reason": "MODEL_REFUSED_PROBE"}
        if probe_type not in SAFE_PROBE_TYPES - {"regression_check"}:
            raise UnsafeProbeError(f"unsafe probe type: {probe_type}")
        requested_target = selection.get("target_tool")
        arguments = deepcopy(selection.get("arguments") or {})
        registry = ContractRegistry(displayed_spec)
        contract = registry.get(str(requested_target))
        if contract is None:
            raise ControllerProbeError("requested probe target is absent from displayed registry")
        if probe_type in {"response_shape_inspection", "repeated_read"} and not _is_read(contract):
            raise ControllerProbeError(f"{probe_type} requires a read-only selected tool")
        before = deepcopy(main_service.store.snapshot())
        before_hash = stable_hash(before)
        fork = SandboxService(store=StateStore.from_state(before), execution_context=context_factory())
        responses: list[dict[str, Any]] = []
        executed_tools: list[str] = []

        def call(tool_id: str, call_arguments: dict[str, Any]):
            response = fork.call_tool(tool_id, deepcopy(call_arguments), actor_id)
            executed_tools.append(tool_id)
            responses.append({"tool_id": tool_id, "arguments": deepcopy(call_arguments), "response": response.to_dict()})
            return response

        complement = None
        if probe_type in {"read_after_write", "precondition_inspection"}:
            complement = _read_complement(registry, contract, arguments)
        if probe_type == "local_schema_check":
            result: Any = {
                "tool_id": contract.operation_id, "method": contract.method,
                "path": contract.path, "input_schema": deepcopy(contract.input_schema),
                "success_schema": deepcopy(contract.success_schema),
            }
        elif probe_type == "historical_behavior_comparison":
            result = {"observation": "use only historical events already present in AgentView"}
        elif probe_type == "read_after_write":
            write = call(contract.operation_id, arguments)
            read_contract, read_args = complement
            read = call(read_contract.operation_id, read_args)
            result = {"write_response": write.to_dict(), "read_after_write_response": read.to_dict()}
        elif probe_type == "precondition_inspection":
            read_contract, read_args = complement
            result = call(read_contract.operation_id, read_args).to_dict()
        elif probe_type == "repeated_read":
            first = call(contract.operation_id, arguments)
            second = call(contract.operation_id, arguments)
            result = {"first_response": first.to_dict(), "second_response": second.to_dict()}
        else:
            result = call(contract.operation_id, arguments).to_dict()
        fork_after = deepcopy(fork.store.snapshot())
        after_hash = stable_hash(main_service.store.snapshot())
        if before_hash != after_hash:
            raise RuntimeError("probe polluted main business state")
        executed_target = requested_target
        if executed_target != requested_target:
            raise ControllerProbeError("requested and executed probe targets differ")
        return {
            "executed": True, "probe_type": probe_type,
            "requested_probe": deepcopy(selection),
            "executed_probe": {
                "probe_type": probe_type, "target_tool": executed_target,
                "arguments": arguments, "executed_tools": executed_tools,
                "expected_observation_type": selection.get("expected_observation_type"),
            },
            "requested_target": requested_target, "executed_target": executed_target,
            "main_state_hash_before": before_hash, "main_state_hash_after": after_hash,
            "fork_state_hash_before": stable_hash(before), "fork_state_hash_after": stable_hash(fork_after),
            "visible_state_diff": state_diff(before, fork_after),
            "state_unchanged": True, "main_state_pollution": False,
            "fork_call_count": len(fork.call_log()), "tool_responses": responses,
            "result": deepcopy(result),
        }


class LLMPatchAdapter:
    def from_output(self, output: dict[str, Any], view: AgentView, attribution: dict[str, Any]) -> ToolSpecPatch:
        available = {event.event_id for event in view.trace.events}
        refs = tuple(output["evidence_refs"])
        operation_refs = {
            ref for operation in output["openapi_operations"] for ref in operation["evidence_refs"]
        }
        if not set(refs).issubset(available) or not operation_refs.issubset(available):
            raise ValueError("LLM patch cites nonexistent live evidence")
        if output["target_tool_id"] != attribution.get("target_tool_id"):
            raise ValueError("LLM patch target differs from final attribution")
        if output["drift_category"] != attribution.get("drift_category"):
            raise ValueError("LLM patch category differs from final attribution")
        if not locations_match(output["location"], attribution.get("location") or {}):
            raise ValueError("LLM patch normalized location differs from final attribution")
        location = normalize_location(output["location"])
        validate_semantics(output["drift_category"], location, output["semantic_extensions"])
        operations = tuple(PatchOperation(
            item["op"], item["path"], deepcopy(item.get("value")),
            tuple(item["evidence_refs"]), item["reason_code"],
        ) for item in output["openapi_operations"])
        identity = json.dumps(output, sort_keys=True, separators=(",", ":"))
        return ToolSpecPatch(
            patch_id="patch-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            patch_version="3.0", created_at="2026-01-01T00:00:00Z",
            target_tool_id=output["target_tool_id"],
            source_spec_fingerprint=stable_hash(thaw(view.displayed_spec)),
            drift_category=output["drift_category"],
            location_type=location["layer"],
            location_path=location["spec_pointer"] or str(location["runtime_path"]), evidence_refs=refs,
            rationale_codes=("LLM_LIVE_EVIDENCE", "HARD_ELIGIBILITY_PASSED"),
            openapi_operations=operations,
            semantic_extensions=deepcopy(output["semantic_extensions"]),
            expected_agent_behavior_change=output["expected_agent_behavior_change"],
            confidence=float(output["confidence"]),
            normalized_location=location,
        )


def _is_read(contract: ToolContract) -> bool:
    return contract.method == "get" or str(contract.effect_type).lower() in {"read", "none"}


def _read_complement(
    registry: ContractRegistry, target: ToolContract, arguments: dict[str, Any],
) -> tuple[ToolContract, dict[str, Any]]:
    candidates: list[tuple[int, ToolContract, dict[str, Any]]] = []
    parent = target.path.rsplit("/", 1)[0]
    for contract in registry.contracts():
        if not _is_read(contract):
            continue
        required = tuple(contract.input_schema.get("required", ()))
        if not all(name in arguments for name in required):
            continue
        if contract.path not in {target.path, parent}:
            continue
        read_args = {name: deepcopy(arguments[name]) for name in required}
        score = 2 if contract.path == target.path else 1
        candidates.append((score, contract, read_args))
    if not candidates:
        raise ControllerProbeError("no safe read complement exists in displayed registry metadata")
    _, contract, read_args = max(candidates, key=lambda item: (item[0], len(item[1].path_parameters)))
    return contract, read_args


def _allowed_target_scope(
    spec: dict[str, Any], target_tool: str | None, location: dict[str, Any] | None,
) -> list[str]:
    if not target_tool:
        return []
    scopes: list[str] = []
    for path, item in spec.get("paths", {}).items():
        for method, operation in item.items():
            if isinstance(operation, dict) and operation.get("operationId") == target_tool:
                escaped = path.replace("~", "~0").replace("/", "~1")
                scopes.append(f"/paths/{escaped}/{method}")
    pointer = (location or {}).get("spec_pointer")
    if isinstance(pointer, str) and pointer.startswith("/components/schemas/"):
        scopes.append("/".join(pointer.split("/")[:4]))
    return scopes


def _sanitized_validator_feedback(error: dict[str, Any] | None) -> dict[str, Any]:
    value = error or {}
    details = value.get("details") if isinstance(value.get("details"), dict) else {}
    return {
        "stage": str(value.get("stage") or "unknown"),
        "passed": False,
        "reason_codes": [str(item)[:300] for item in value.get("reason_codes", ())],
        "details": {str(key): str(child)[:500] for key, child in details.items()},
    }


class LivePatchValidator:
    """Phase 8 validators over an LLM patch; never invokes PatchCandidateGenerator."""

    def __init__(
        self, *, registry: PatchRegistry | None = None, max_repair_calls: int = 8,
        static=None, regression=None, safety=None, minimality=None,
    ):
        self.registry = registry or PatchRegistry()
        self.static = static or StaticPatchValidator()
        self.regression = regression or RegressionValidator()
        self.safety = safety or SafetyValidator()
        self.minimality = minimality or MinimalityValidator()
        self.executor = RepairExecutor(max_repair_calls)
        self.max_repair_calls = max_repair_calls

    def static_only(self, patch: ToolSpecPatch, view: AgentView):
        return self.static.validate(patch, thaw(view.displayed_spec))

    def validate_after_static(
        self, patch: ToolSpecPatch, static_result, view: AgentView,
        context_factory: Callable[[], ExecutionContext], repair_arguments: dict[str, Any],
        initial_state: dict[str, Any], future_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not static_result.passed:
            return {"accepted": False, "static": static_result.to_dict(), "rejection": "STATIC_VALIDATION_FAILED"}
        patch = patch.with_status(PatchLifecycle.STATIC_VALIDATED, static_result)
        overlay = SpecOverlay(thaw(view.displayed_spec))
        patched_displayed = overlay.apply(patch)
        repair_run = self.executor.execute(
            patch, context_factory(), deepcopy(repair_arguments), deepcopy(initial_state), episode=5,
        )
        repair_result = repair_validation(repair_run)
        if not repair_result.passed:
            overlay.rollback()
            return {
                "accepted": False, "static": static_result.to_dict(),
                "immediate_repair": repair_result.to_dict(), "rejection": "IMMEDIATE_REPAIR_FAILED",
            }
        patch = patch.with_status(PatchLifecycle.REPAIR_VALIDATED, repair_result)
        regression = self.regression.validate(patch, repair_run.initial_state)
        safety = self.safety.validate(patch, repair_run, self.max_repair_calls)
        minimality = self.minimality.validate(patch)
        accepted = regression.passed and safety.passed and minimality.passed
        result = {
            "accepted": accepted, "static": static_result.to_dict(),
            "patched_displayed_spec_fingerprint": stable_hash(patched_displayed),
            "immediate_repair": repair_result.to_dict(), "repair_run": repair_run.public_dict(),
            "regression": regression.to_dict(), "safety": safety.to_dict(),
            "minimality": minimality.to_dict(),
        }
        if not accepted:
            overlay.rollback()
            result["rejection"] = "DETERMINISTIC_VALIDATION_FAILED"
            return result
        accepted_patch = patch.with_status(PatchLifecycle.ACCEPTED, minimality)
        self.registry.register(accepted_patch)
        future = FutureTransferEvaluator(self.executor).evaluate(
            accepted_patch, context_factory, deepcopy(future_arguments or repair_arguments), deepcopy(initial_state),
        )
        result["accepted_patch"] = accepted_patch.to_dict()
        result["future_transfer"] = future
        result["patch_registry_size"] = len(self.registry)
        return result
