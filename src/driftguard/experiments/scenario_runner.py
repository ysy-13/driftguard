from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from driftguard.agents import ToolAgentController
from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.agents.policies import POLICIES
from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.evidence.models import thaw
from driftguard.experiments.budgets import BudgetTracker
from driftguard.healing import HealingEngine
from driftguard.llm import InvalidStructuredOutput, LLMCache, MockProvider, ModelConfig, ProviderRequest, parse_structured_output
from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner
from driftguard.runners.healing_conformance_runner import HealingConformanceRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService

from .config import ExperimentConfig
from .task_resolver import MATCHED_PATH, TaskInstanceResolver


LABEL_MAP = {"agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"}


class ExperimentScenarioRunner:
    def __init__(self, config: ExperimentConfig, cache: LLMCache | None, prompts: dict[str, str], schemas: dict[str, dict], provider_factory=None):
        self.config, self.cache, self.prompts, self.schemas = config, cache, prompts, schemas
        self.attribution = AttributionConformanceRunner()
        self.healing_runner = HealingConformanceRunner()
        self._artifact_cache: dict[str, dict[str, Any]] = {}
        self._healing_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self.provider_factory = provider_factory or (lambda outputs: MockProvider(outputs, self.config.model.model_id))
        self.task_resolver = TaskInstanceResolver()
        self.catalog_renderer = ToolCatalogRenderer()

    def artifacts(self, family: dict, scenario: dict) -> dict[str, Any]:
        key = scenario["scenario_id"]
        with self._cache_lock:
            if key not in self._artifact_cache:
                self._artifact_cache[key] = self.attribution._run_scenario(family, scenario, return_agent_artifacts=True)
            return self._artifact_cache[key]

    def run(self, family: dict, scenario: dict, method: str, mode: str, repetition: int) -> dict[str, Any]:
        if mode == "component":
            return self._component(family, scenario, method, repetition)
        return self._end_to_end(family, scenario, method, repetition)

    def healing_artifact(self, family: dict, scenario: dict) -> dict[str, Any]:
        key = scenario["scenario_id"]
        with self._cache_lock:
            if key not in self._healing_cache:
                self._healing_cache[key] = self.healing_runner._run_once(family, scenario)
            return self._healing_cache[key]

    def _component(self, family: dict, scenario: dict, method: str, repetition: int) -> dict[str, Any]:
        artifacts = self.artifacts(family, scenario)
        case = self.attribution._case_by_id[family["source_drift_id"]]
        diagnosis = artifacts["diagnosis"]
        expected = LABEL_MAP[scenario["evaluator_metadata"]["ground_truth_label"]]
        predicted, localization = None, None
        patch_proposed = patch_accepted = False
        immediate = transfer = None
        raw_patch_correct = accepted_patch_correct = regression_pass = minimality_pass = None
        llm_calls = input_tokens = output_tokens = 0
        if method in {"driftguard_symbolic", "oracle_symbolic_upper_bound"}:
            predicted, localization = diagnosis.predicted_class, diagnosis.localization.to_dict()
            healed = self.healing_artifact(family, scenario)
            patch_proposed = bool(healed["proposed_candidates"])
            patch_accepted = bool(healed["acceptance"])
            immediate = bool(healed["immediate_repair"] and healed["immediate_repair"]["passed"])
            transfer = bool(healed["future_transfer"] and healed["future_transfer"]["patched_success"])
            raw_patch_correct = bool(healed["evaluator_correctness"]["semantic_patch"]) if patch_proposed else None
            accepted_patch_correct = raw_patch_correct if patch_accepted else None
            regression_pass = healed["validation"].get("regression", {}).get("passed")
            minimality_pass = healed["validation"].get("minimality", {}).get("passed")
        else:
            output = {
                "predicted_class": "INSUFFICIENT_EVIDENCE", "target_tool_id": None,
                "drift_category": "UNKNOWN", "location_type": "unknown", "location_path": None,
                "evidence_refs": [], "requested_probe": None, "confidence": 0.0,
                "concise_reason": "mock component abstention",
            }
            provider = self.provider_factory([output])
            visible_evidence = {
                "displayed_spec": thaw(artifacts["agent_view"].displayed_spec),
                "evidence_trace": artifacts["agent_view"].trace.to_dict(),
                "current_episode": artifacts["agent_view"].current_episode,
            }
            component_v2 = self.prompts.get("component_attribution_v2.txt")
            output_schema_text = json.dumps(self.schemas["llm_attribution"], indent=2, sort_keys=True)
            system_prompt = (
                component_v2.replace("{{OUTPUT_SCHEMA}}", output_schema_text) + "\n" + self._policy_prompt(method)
                if component_v2 is not None
                else self.prompts["base_tool_agent_v1.txt"] + "\n" + self._policy_prompt(method)
            )
            component_prompt_hash = self._component_prompt_hash(method)
            messages = (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Analyze this Agent-visible evidence and return only the requested attribution JSON:\n" + json.dumps(visible_evidence, sort_keys=True)},
            )
            schema_hash = hashlib.sha256(json.dumps(self.schemas["llm_attribution"], sort_keys=True).encode()).hexdigest()
            parsed = None
            responses = []
            component_tracker = BudgetTracker(self.config.budget)
            for repair_index in range(self.config.budget.max_format_repairs + 1):
                request = ProviderRequest(
                    messages, self.schemas["llm_attribution"], component_prompt_hash,
                    artifacts["public_scenario_id"], 5, method, repetition,
                    self.config.model.seed, "component", self.config.config_hash,
                )
                cache_key = LLMCache.key(self.config.model, request, schema_hash) if self.cache else None
                response = (
                    self.cache.get(cache_key)
                    if self.cache and cache_key and not self.config.cache.get("force_refresh", False)
                    else None
                )
                if response is None:
                    component_tracker.ensure_llm_call_allowed()
                    response = provider.complete(request)
                    if self.cache and cache_key:
                        self.cache.put(cache_key, response)
                else:
                    component_tracker.ensure_llm_call_allowed()
                responses.append(response)
                component_tracker.consume_llm(response.input_tokens, response.output_tokens)
                try:
                    parsed = parse_structured_output(response.raw_text, self.schemas["llm_attribution"])
                    break
                except InvalidStructuredOutput as exc:
                    if repair_index >= self.config.budget.max_format_repairs:
                        raise
                    component_tracker.consume_format_repair()
                    messages = messages + ({
                        "role": "user", "content": (
                            "Schema validation error: " + str(exc) + "\n"
                            "Return only one corrected JSON object. The complete output schema is:\n"
                            "<OUTPUT_SCHEMA>\n" + output_schema_text + "\n</OUTPUT_SCHEMA>"
                        ),
                    },)
            if parsed is None:
                raise InvalidStructuredOutput("component attribution repair failed")
            predicted = {
                "AGENT_ERROR": "AE", "TRANSIENT_FAILURE": "TF",
                "PERSISTENT_DRIFT": "PD",
            }.get(parsed["predicted_class"], parsed["predicted_class"])
            localization = {
                key: parsed[key] for key in ("target_tool_id", "drift_category", "location_type", "location_path")
            }
            llm_calls = component_tracker.llm_calls
            input_tokens = component_tracker.input_tokens
            output_tokens = component_tracker.output_tokens
        record = self._record(
            artifacts["public_scenario_id"], mode="component", method=method, repetition=repetition,
            success=predicted == expected, predicted=predicted, localization=localization,
            patch_proposed=patch_proposed, patch_accepted=patch_accepted, immediate=immediate, transfer=transfer,
            raw_patch_correct=raw_patch_correct, accepted_patch_correct=accepted_patch_correct,
            regression_pass=regression_pass, minimality_pass=minimality_pass,
            llm_calls=llm_calls, input_tokens=input_tokens, output_tokens=output_tokens,
            termination="COMPONENT_COMPLETE", prompt_hash=(self._component_prompt_hash(method)), expected=expected,
            expected_tool=case["target_tool"], expected_category=_category(case["drift_type"]),
            expected_location=case["ground_truth"]["location"],
        )
        record["_actual_models"] = [item.model for item in responses] if method not in {"driftguard_symbolic", "oracle_symbolic_upper_bound"} else []
        record["_provider_attempts"] = sum(item.provider_attempts for item in responses) if method not in {"driftguard_symbolic", "oracle_symbolic_upper_bound"} else 0
        return record

    def _end_to_end(self, family: dict, scenario: dict, method: str, repetition: int) -> dict[str, Any]:
        case = self.attribution._case_by_id[family["source_drift_id"]]
        mode = ExecutionMode(scenario["variant_code"])
        context = ExecutionContext(ExecutionProfile(mode, case, family, change_point=int(case["change_point"])))
        context.set_episode(3)
        service = SandboxService(execution_context=context)
        state_isolation_valid = service.store.snapshot() == self.task_resolver.fixture
        policy = POLICIES[method]()
        resolved = self.task_resolver.resolve(family, scenario)
        task = resolved.evaluator_task()
        first_step = next(step for step in resolved.oracle_plan if not step.get("when"))
        action = {
            "action_type": "TOOL_CALL", "tool_id": first_step["tool"],
            "arguments": deepcopy(first_step["arguments"]),
            "concise_decision_summary": "execute the visible task with the displayed tool",
        }
        provider = self.provider_factory([action, {
            "action_type": "FINAL_ANSWER", "answer": "unable to complete after visible failure",
            "concise_decision_summary": "stop after bounded recovery",
        }])
        tracker = BudgetTracker(self.config.budget)
        action_schema = self._runtime_action_schema(method)
        controller = ToolAgentController(
            provider, self.config.model, policy, service, tracker, action_schema,
            self._action_base_prompt(method, action_schema), self._policy_prompt(method), self.cache,
            bool(self.config.cache.get("force_refresh", False)),
            self.config.config_hash, "end_to_end",
            bool(self.config.raw.get("execution", {}).get("replay_normalized_runtime_arguments", False)),
        )
        run = controller.run(task, context.displayed_contract, _public_id(scenario["scenario_id"]), 3, repetition)
        predicted, localization = None, None
        patch_proposed = patch_accepted = False
        immediate = transfer = None
        raw_patch_correct = accepted_patch_correct = regression_pass = minimality_pass = None
        diagnostic_llm_calls = diagnostic_input_tokens = diagnostic_output_tokens = 0
        diagnostic_actual_models: list[str] = []
        diagnostic_provider_attempts = 0
        if method in {"driftguard_symbolic", "oracle_symbolic_upper_bound"}:
            artifacts = self.artifacts(family, scenario)
            diagnosis = artifacts["diagnosis"]
            predicted, localization = diagnosis.predicted_class, diagnosis.localization.to_dict()
            healed = self.healing_artifact(family, scenario)
            patch_proposed, patch_accepted = bool(healed["proposed_candidates"]), bool(healed["acceptance"])
            immediate = bool(healed["immediate_repair"] and healed["immediate_repair"]["passed"])
            transfer = bool(healed["future_transfer"] and healed["future_transfer"]["patched_success"])
            raw_patch_correct = bool(healed["evaluator_correctness"]["semantic_patch"]) if patch_proposed else None
            accepted_patch_correct = raw_patch_correct if patch_accepted else None
            regression_pass = healed["validation"].get("regression", {}).get("passed")
            minimality_pass = healed["validation"].get("minimality", {}).get("passed")
        elif method == "driftguard_llm":
            # Attribution is another LLM decision over the frozen visible
            # evidence. It is never replaced with the symbolic diagnosis.
            diagnostic = self._component(family, scenario, method, repetition)
            predicted, localization = diagnostic["predicted_class"], diagnostic["localization"]
            diagnostic_llm_calls = diagnostic["llm_calls"]
            diagnostic_input_tokens = diagnostic["input_tokens"]
            diagnostic_output_tokens = diagnostic["output_tokens"]
            diagnostic_actual_models = diagnostic.get("_actual_models", [])
            diagnostic_provider_attempts = diagnostic.get("_provider_attempts", 0)
        expected = LABEL_MAP[scenario["evaluator_metadata"]["ground_truth_label"]]
        record = self._record(
            _public_id(scenario["scenario_id"]), mode="end_to_end", method=method, repetition=repetition,
            success=run.task_success, predicted=predicted, localization=localization,
            patch_proposed=patch_proposed, patch_accepted=patch_accepted, immediate=immediate, transfer=transfer,
            raw_patch_correct=raw_patch_correct, accepted_patch_correct=accepted_patch_correct,
            regression_pass=regression_pass, minimality_pass=minimality_pass,
            llm_calls=run.llm_calls + diagnostic_llm_calls, tool_calls=run.tool_calls, probe_calls=run.probe_calls,
            input_tokens=run.input_tokens + diagnostic_input_tokens,
            output_tokens=run.output_tokens + diagnostic_output_tokens, latency=run.latency_ms,
            termination=run.termination_reason, error=run.error_category,
            prompt_hash=self._prompt_hash(method, context.displayed_contract), trace_refs=list(run.trace_references), expected=expected,
            catalog_fingerprint=run.catalog_fingerprint,
            forbidden=0 if run.task_evaluation.get("forbidden_assertions_passed", False) else 1,
            expected_tool=case["target_tool"], expected_category=_category(case["drift_type"]),
            expected_location=case["ground_truth"]["location"],
        )
        record["_actual_models"] = list(dict.fromkeys(controller.usage.actual_models + diagnostic_actual_models))
        record["_provider_attempts"] = controller.usage.provider_attempts + diagnostic_provider_attempts
        record["_policy_events"] = list(run.policy_events)
        record["_state_transitions"] = list(run.state_transitions)
        record["_actions"] = list(run.actions)
        record["_task_evaluation"] = dict(run.task_evaluation)
        record["_state_isolation_valid"] = state_isolation_valid
        record["_driftguard_path"] = {
            "evidence_entered": any(event.get("method", "").startswith("driftguard_") for event in run.policy_events),
            "attribution_entered": method == "driftguard_llm",
            "probe_entered": run.probe_calls > 0,
            "eligibility_entered": method == "driftguard_llm" and predicted is not None,
            "healing_entered": method == "driftguard_llm" and predicted is not None,
            "healing_outcome": "NO_ACCEPTED_PATCH" if method == "driftguard_llm" else None,
        }
        return record

    def _record(
        self, public_id, *, mode, method, repetition, success, predicted, localization,
        patch_proposed, patch_accepted, immediate, transfer,
        raw_patch_correct=None, accepted_patch_correct=None, regression_pass=None, minimality_pass=None,
        llm_calls=0, tool_calls=0,
        probe_calls=0, input_tokens=0, output_tokens=0, latency=0.0,
        termination, error=None, prompt_hash, trace_refs=None, expected, forbidden=0,
        expected_tool=None, expected_category=None, expected_location=None, catalog_fingerprint="",
    ):
        false_patch = bool(patch_proposed and expected != "PD")
        target_correct = None if localization is None else localization.get("target_tool_id", localization.get("tool_id")) == expected_tool
        category_correct = None if localization is None else localization.get("drift_category") == expected_category
        location_correct = None if localization is None else localization.get("location_path") == expected_location
        return {
            "experiment_id": self.config.experiment_id, "public_scenario_id": public_id,
            "mode": mode, "method": method, "model": self.config.model.model_id,
            "repetition": repetition, "final_task_success": bool(success),
            "predicted_class": predicted, "localization": localization,
            "target_tool_correct": target_correct,
            "drift_category_correct": category_correct,
            "exact_location_correct": location_correct,
            "patch_proposed": bool(patch_proposed), "patch_accepted": bool(patch_accepted),
            "raw_patch_correct": raw_patch_correct, "accepted_patch_correct": accepted_patch_correct,
            "regression_pass": regression_pass, "minimality_pass": minimality_pass, "unsafe_patch": False,
            "immediate_repair": immediate, "future_transfer": transfer, "false_patch": false_patch,
            "unsafe_action": False, "forbidden_side_effects": forbidden,
            "llm_calls": llm_calls, "tool_calls": tool_calls, "probe_calls": probe_calls,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "latency_ms": latency,
            "termination_reason": termination, "error_category": error,
            "prompt_hash": prompt_hash, "trace_references": trace_refs or [], "mock_provider": self.config.model.provider == "mock",
            "catalog_fingerprint": catalog_fingerprint,
            "_expected": expected,
        }

    def _policy_prompt(self, method: str) -> str:
        names = {
            "reflection": "reflection_v1.txt", "validation_guided": "validation_guided_v1.txt",
            "driftguard_symbolic": "driftguard_attribution_v1.txt", "driftguard_llm": "driftguard_attribution_v1.txt",
            "oracle_symbolic_upper_bound": "driftguard_attribution_v1.txt",
        }
        return self.prompts.get(names.get(method, ""), "")

    def _prompt_hash(self, method: str, displayed_spec: dict[str, Any] | None = None) -> str:
        if not self._uses_action_v3():
            return hashlib.sha256(self._action_system_prompt(method).encode()).hexdigest()
        capabilities = PolicyCapabilities.for_method(method)
        schema = self._runtime_action_schema(method)
        material = {
            "system_prompt": self._action_system_prompt(method, schema),
            "action_schema": schema,
            "policy_capabilities": {"method": method, "allowed_actions": capabilities.allowed_actions},
            "method_policy": self._policy_prompt(method),
            "tool_catalog": None if displayed_spec is None else self.catalog_renderer.render(displayed_spec),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _action_base_prompt(self, method: str = "standard", action_schema: dict[str, Any] | None = None) -> str:
        requested = self.config.raw.get("experiment", {}).get("action_prompt_version")
        template = self.prompts.get(f"{requested}.txt") if requested else None
        template = template or self.prompts.get("base_tool_agent_v2.txt")
        if template is None:
            return self.prompts["base_tool_agent_v1.txt"]
        capabilities = PolicyCapabilities.for_method(method)
        action_schema = action_schema or self._runtime_action_schema(method)
        schema = json.dumps(action_schema, indent=2, sort_keys=True)
        return template.replace("{{OUTPUT_SCHEMA}}", schema).replace("{{POLICY_CAPABILITIES}}", capabilities.prompt_fragment())

    def _action_system_prompt(self, method: str, action_schema: dict[str, Any] | None = None) -> str:
        return self._action_base_prompt(method, action_schema) + "\n" + self._policy_prompt(method)

    def _uses_action_v3(self) -> bool:
        return self.config.raw.get("experiment", {}).get("action_prompt_version") == "base_tool_agent_v3"

    def _runtime_action_schema(self, method: str) -> dict[str, Any]:
        if self._uses_action_v3():
            return PolicyCapabilities.for_method(method).narrow_schema(self.schemas["agent_action"])
        return deepcopy(self.schemas["agent_action"])

    def _component_prompt_hash(self, method: str) -> str:
        template = self.prompts.get("component_attribution_v2.txt")
        if template is None:
            return self._prompt_hash(method)
        schema = json.dumps(self.schemas["llm_attribution"], indent=2, sort_keys=True)
        rendered = template.replace("{{OUTPUT_SCHEMA}}", schema) + "\n" + self._policy_prompt(method)
        return hashlib.sha256(rendered.encode()).hexdigest()


def _public_id(scenario_id: str) -> str:
    return f"public-{hashlib.sha256(scenario_id.encode()).hexdigest()[:12]}"


def _smoke_task(case: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))
    family = next(item for item in document["families"] if item["source_drift_id"] == case["drift_id"])
    return TaskInstanceResolver().resolve(family, scenario).evaluator_task()


def _legacy_smoke_task(case: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Frozen adapter used only to reproduce the already-recorded Pilot v3 audit."""
    from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS

    repo_id = CASE_ARGUMENTS[case["drift_id"]].get("repo_id", "R1")
    return {
        "task_id": "phase9-mock-smoke", "instruction": scenario["agent_visible"]["evidence_bundle"]["user_task"],
        "actor_id": "agent_admin", "success_assertions": [],
        "answer_assertions": [{"key": "tool_succeeded", "operator": "eq", "value": True}],
        "forbidden_assertions": [{"allowed_paths": [f"/repositories/{repo_id}/**", f"/next_ids/{repo_id}/**"]}],
        "max_tool_calls": 6,
    }


def _category(drift_type: str) -> str:
    return {
        "input_contract": "ICD", "response_shape": "RSD",
        "workflow_precondition": "WPD", "state_effect": "SED",
    }[drift_type]
