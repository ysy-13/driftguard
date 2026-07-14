from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from driftguard.agents import ToolAgentController
from driftguard.agents.policies import POLICIES
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.budgets import BudgetTracker
from driftguard.healing import HealingEngine
from driftguard.llm import LLMCache, MockProvider, ModelConfig, ProviderRequest, parse_structured_output
from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner
from driftguard.runners.healing_conformance_runner import HealingConformanceRunner
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService

from .config import ExperimentConfig


LABEL_MAP = {"agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"}


class ExperimentScenarioRunner:
    def __init__(self, config: ExperimentConfig, cache: LLMCache | None, prompts: dict[str, str], schemas: dict[str, dict], provider_factory=None):
        self.config, self.cache, self.prompts, self.schemas = config, cache, prompts, schemas
        self.attribution = AttributionConformanceRunner()
        self.healing_runner = HealingConformanceRunner()
        self._artifact_cache: dict[str, dict[str, Any]] = {}
        self._healing_cache: dict[str, dict[str, Any]] = {}
        self.provider_factory = provider_factory or (lambda outputs: MockProvider(outputs, self.config.model.model_id))

    def artifacts(self, family: dict, scenario: dict) -> dict[str, Any]:
        key = scenario["scenario_id"]
        if key not in self._artifact_cache:
            self._artifact_cache[key] = self.attribution._run_scenario(family, scenario, return_agent_artifacts=True)
        return self._artifact_cache[key]

    def run(self, family: dict, scenario: dict, method: str, mode: str, repetition: int) -> dict[str, Any]:
        if mode == "component":
            return self._component(family, scenario, method, repetition)
        return self._end_to_end(family, scenario, method, repetition)

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
            key = scenario["scenario_id"]
            if key not in self._healing_cache:
                self._healing_cache[key] = self.healing_runner._run_once(family, scenario)
            healed = self._healing_cache[key]
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
            request = ProviderRequest(
                ({"role": "system", "content": self.prompts["base_tool_agent_v1.txt"]},),
                self.schemas["llm_attribution"], self._prompt_hash(method),
                artifacts["public_scenario_id"], 5, method, repetition,
            )
            schema_hash = hashlib.sha256(json.dumps(self.schemas["llm_attribution"], sort_keys=True).encode()).hexdigest()
            cache_key = LLMCache.key(self.config.model, request, schema_hash) if self.cache else None
            response = (
                self.cache.get(cache_key)
                if self.cache and cache_key and not self.config.cache.get("force_refresh", False)
                else None
            )
            if response is None:
                response = provider.complete(request)
                if self.cache and cache_key:
                    self.cache.put(cache_key, response)
            parsed = parse_structured_output(response.raw_text, self.schemas["llm_attribution"])
            predicted = parsed["predicted_class"]
            localization = {
                key: parsed[key] for key in ("target_tool_id", "drift_category", "location_type", "location_path")
            }
            llm_calls, input_tokens, output_tokens = 1, response.input_tokens, response.output_tokens
        return self._record(
            artifacts["public_scenario_id"], mode="component", method=method, repetition=repetition,
            success=predicted == expected, predicted=predicted, localization=localization,
            patch_proposed=patch_proposed, patch_accepted=patch_accepted, immediate=immediate, transfer=transfer,
            raw_patch_correct=raw_patch_correct, accepted_patch_correct=accepted_patch_correct,
            regression_pass=regression_pass, minimality_pass=minimality_pass,
            llm_calls=llm_calls, input_tokens=input_tokens, output_tokens=output_tokens,
            termination="COMPONENT_COMPLETE", prompt_hash=self._prompt_hash(method), expected=expected,
            expected_tool=case["target_tool"], expected_category=_category(case["drift_type"]),
            expected_location=case["ground_truth"]["location"],
        )

    def _end_to_end(self, family: dict, scenario: dict, method: str, repetition: int) -> dict[str, Any]:
        case = self.attribution._case_by_id[family["source_drift_id"]]
        mode = ExecutionMode(scenario["variant_code"])
        context = ExecutionContext(ExecutionProfile(mode, case, family, change_point=int(case["change_point"])))
        context.set_episode(3)
        service = SandboxService(execution_context=context)
        policy = POLICIES[method]()
        action = {
            "action_type": "TOOL_CALL", "tool_id": case["target_tool"],
            "arguments": deepcopy(CASE_ARGUMENTS[case["drift_id"]]),
            "concise_decision_summary": "execute the visible task with the displayed tool",
        }
        provider = self.provider_factory([action, {
            "action_type": "FINAL_ANSWER", "answer": "unable to complete after visible failure",
            "concise_decision_summary": "stop after bounded recovery",
        }])
        tracker = BudgetTracker(self.config.budget)
        controller = ToolAgentController(
            provider, self.config.model, policy, service, tracker, self.schemas["agent_action"],
            self.prompts["base_tool_agent_v1.txt"], self._policy_prompt(method), self.cache,
            bool(self.config.cache.get("force_refresh", False)),
        )
        task = _smoke_task(case, scenario)
        run = controller.run(task, context.displayed_contract, _public_id(scenario["scenario_id"]), 3, repetition)
        predicted, localization = None, None
        patch_proposed = patch_accepted = False
        immediate = transfer = None
        raw_patch_correct = accepted_patch_correct = regression_pass = minimality_pass = None
        if method in {"driftguard_symbolic", "oracle_symbolic_upper_bound"}:
            artifacts = self.artifacts(family, scenario)
            diagnosis = artifacts["diagnosis"]
            predicted, localization = diagnosis.predicted_class, diagnosis.localization.to_dict()
            key = scenario["scenario_id"]
            if key not in self._healing_cache:
                self._healing_cache[key] = self.healing_runner._run_once(family, scenario)
            healed = self._healing_cache[key]
            patch_proposed, patch_accepted = bool(healed["proposed_candidates"]), bool(healed["acceptance"])
            immediate = bool(healed["immediate_repair"] and healed["immediate_repair"]["passed"])
            transfer = bool(healed["future_transfer"] and healed["future_transfer"]["patched_success"])
            raw_patch_correct = bool(healed["evaluator_correctness"]["semantic_patch"]) if patch_proposed else None
            accepted_patch_correct = raw_patch_correct if patch_accepted else None
            regression_pass = healed["validation"].get("regression", {}).get("passed")
            minimality_pass = healed["validation"].get("minimality", {}).get("passed")
        elif method == "driftguard_llm":
            # The mock LLM abstains; no symbolic result is substituted.
            predicted = "INSUFFICIENT_EVIDENCE"
            localization = {"target_tool_id": None, "drift_category": "UNKNOWN", "location_type": "unknown", "location_path": None}
        expected = LABEL_MAP[scenario["evaluator_metadata"]["ground_truth_label"]]
        record = self._record(
            _public_id(scenario["scenario_id"]), mode="end_to_end", method=method, repetition=repetition,
            success=run.task_success, predicted=predicted, localization=localization,
            patch_proposed=patch_proposed, patch_accepted=patch_accepted, immediate=immediate, transfer=transfer,
            raw_patch_correct=raw_patch_correct, accepted_patch_correct=accepted_patch_correct,
            regression_pass=regression_pass, minimality_pass=minimality_pass,
            llm_calls=run.llm_calls, tool_calls=run.tool_calls, probe_calls=run.probe_calls,
            input_tokens=run.input_tokens, output_tokens=run.output_tokens, latency=run.latency_ms,
            termination=run.termination_reason, error=run.error_category,
            prompt_hash=self._prompt_hash(method), trace_refs=list(run.trace_references), expected=expected,
            forbidden=0 if run.task_evaluation.get("forbidden_assertions_passed", False) else 1,
            expected_tool=case["target_tool"], expected_category=_category(case["drift_type"]),
            expected_location=case["ground_truth"]["location"],
        )
        return record

    def _record(
        self, public_id, *, mode, method, repetition, success, predicted, localization,
        patch_proposed, patch_accepted, immediate, transfer,
        raw_patch_correct=None, accepted_patch_correct=None, regression_pass=None, minimality_pass=None,
        llm_calls=0, tool_calls=0,
        probe_calls=0, input_tokens=0, output_tokens=0, latency=0.0,
        termination, error=None, prompt_hash, trace_refs=None, expected, forbidden=0,
        expected_tool=None, expected_category=None, expected_location=None,
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
            "_expected": expected,
        }

    def _policy_prompt(self, method: str) -> str:
        names = {
            "reflection": "reflection_v1.txt", "validation_guided": "validation_guided_v1.txt",
            "driftguard_symbolic": "driftguard_attribution_v1.txt", "driftguard_llm": "driftguard_attribution_v1.txt",
            "oracle_symbolic_upper_bound": "driftguard_attribution_v1.txt",
        }
        return self.prompts.get(names.get(method, ""), "")

    def _prompt_hash(self, method: str) -> str:
        return hashlib.sha256((self.prompts["base_tool_agent_v1.txt"] + self._policy_prompt(method)).encode()).hexdigest()


def _public_id(scenario_id: str) -> str:
    return f"public-{hashlib.sha256(scenario_id.encode()).hexdigest()[:12]}"


def _smoke_task(case: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
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
