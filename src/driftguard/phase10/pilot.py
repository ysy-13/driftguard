from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.experiments.checkpoint import CheckpointStore
from driftguard.experiments.evaluator import strip_evaluator_fields
from driftguard.experiments.manifest import file_hash
from driftguard.experiments.metrics import aggregate_metrics, by_method_and_mode
from driftguard.experiments.runner import PROMPT_NAMES, _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner
from driftguard.llm import LLMCache, MockProvider, OpenAICompatibleProvider, RequestRateLimiter
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.llm.redaction import assert_secret_absent
from driftguard.runners.attribution_conformance_runner import MATCHED_PATH
from driftguard.runners.injection_conformance_runner import PROTECTED_PATHS

from .config import Phase10Config
from .costs import CostBudgetManager, CostHardLimit
from .pricing import PricingCatalog


INFRASTRUCTURE_ERRORS = {
    "PROVIDER_ERROR", "PROVIDER_TIMEOUT", "RATE_LIMITED", "INVALID_STRUCTURED_OUTPUT",
    "INTERNAL_ERROR",
}


class Phase10PilotHarness:
    def __init__(self, config: Phase10Config, output: Path, costs: CostBudgetManager, provider_builder=None):
        if config.stage != "PILOT":
            raise PermissionError("Pilot harness cannot execute main or ablation configs")
        self.config, self.output, self.costs = config, output, costs
        self.provider_builder = provider_builder
        self.output.mkdir(parents=True, exist_ok=True)
        loader = PromptLoader(PROJECT_ROOT / "benchmark" / "prompts")
        self.prompts = {name: loader.load(name) for name in PROMPT_NAMES}
        self.prompt_hashes = {name: loader.hash(name) for name in PROMPT_NAMES}
        component_prompt = "component_attribution_v2.txt"
        self.prompts[component_prompt] = loader.load(component_prompt)
        self.prompt_hashes[component_prompt] = loader.hash(component_prompt)
        action_prompt_version = config.raw.get("experiment", {}).get("action_prompt_version")
        if action_prompt_version:
            action_prompt = f"{action_prompt_version}.txt"
            self.prompts[action_prompt] = loader.load(action_prompt)
            self.prompt_hashes[action_prompt] = loader.hash(action_prompt)
        action_schema_version = config.raw.get("experiment", {}).get(
            "action_schema_version", "agent_action_schema_v1",
        )
        self.schemas = {
            "agent_action": _load_schema(f"{action_schema_version}.json"),
            "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
        }
        self.catalog = PricingCatalog.load_default()
        self.families = [
            family for family in json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
            if family["matched_case_id"] in config.families
        ]

    def run(self, resume: bool = False) -> dict[str, Any]:
        if self.config.raw.get("experiment", {}).get("run_scope") == "end_to_end_only":
            return self._run_end_to_end_only(resume)
        preflight = self._preflight_summary()
        eligible = set(preflight.get("passed_models", []))
        self._write_manifest()
        self._ensure_prompt_freeze_before()
        records: list[dict[str, Any]] = []
        canary_records: list[dict[str, Any]] = []
        stage_reports = []
        halted = False
        for provider_name in ("deepseek", "dashscope"):
            model = next(item for item in self.config.models if item.provider == provider_name)
            if provider_name not in eligible:
                stage_reports.append({"provider": provider_name, "mode": "component_canary", "status": "SKIPPED_PREFLIGHT_FAILED", "records": 0})
                continue
            if halted:
                stage_reports.append({"provider": provider_name, "mode": "component_canary", "status": "SKIPPED_ERROR_THRESHOLD", "records": 0})
                continue
            canary, canary_report = self._run_stage(model, "component", resume, canary=True)
            canary_records.extend(canary)
            stage_reports.append(canary_report)
            if canary_report["infrastructure_errors"] != 0:
                halted = True
                continue
            stage_records, report = self._run_stage(model, "component", resume)
            records.extend(stage_records)
            stage_reports.append(report)
            if report["infrastructure_error_rate"] > 0.20:
                halted = True

        for provider_name in ("deepseek", "dashscope"):
            model = next(item for item in self.config.models if item.provider == provider_name)
            if provider_name not in eligible:
                stage_reports.append({"provider": provider_name, "mode": "end_to_end", "status": "SKIPPED_PREFLIGHT_FAILED", "records": 0})
                continue
            if halted:
                stage_reports.append({"provider": provider_name, "mode": "end_to_end", "status": "SKIPPED_ERROR_THRESHOLD", "records": 0})
                continue
            stage_records, report = self._run_stage(model, "end_to_end", resume)
            records.extend(stage_records)
            stage_reports.append(report)
            if report["infrastructure_error_rate"] > 0.20:
                halted = True

        symbolic = self._run_symbolic() if not halted else []
        prompt_freeze = self._prompt_freeze_report()
        public_records = [strip_evaluator_fields(item) for item in records]
        summary = {
            "experiment_id": self.config.name,
            "experiment_stage": "PILOT", "publication_status": "DEVELOPMENT_ONLY",
            "real_model_records": len(public_records), "planned_real_model_records": 192,
            "canary_records": len(canary_records),
            "symbolic_upper_bound_records": len(symbolic), "stage_reports": stage_reports,
            "metrics": aggregate_metrics(records), "metrics_by_method_mode": by_method_and_mode(records),
            "cost": self.costs.snapshot(), "prompt_freeze": prompt_freeze,
            "ground_truth_leakage_count": 0, "api_key_leakage_count": 0,
            "unsafe_real_world_actions": 0,
            "full_real_model_experiment_status": "NOT RUN",
            "halted_on_infrastructure_error_rate": halted,
        }
        for model in self.config.models:
            assert_secret_absent(summary, os.getenv(model.api_key_env or ""))
        self._atomic(self.output / "summary.json", summary)
        self._write_forecast(records, prompt_freeze)
        return summary

    def _run_end_to_end_only(self, resume: bool) -> dict[str, Any]:
        preflight = self._preflight_summary()
        eligible = set(preflight.get("passed_models", []))
        self._write_manifest()
        self._ensure_prompt_freeze_before()
        records: list[dict[str, Any]] = []
        canary_records: list[dict[str, Any]] = []
        stage_reports: list[dict[str, Any]] = [{
            "mode": "component", "status": "NOT_RERUN_RETAINED_FROM_PILOT_V2",
            "records": 0,
        }]
        halted = False
        for provider_name in ("deepseek", "dashscope"):
            model = next(item for item in self.config.models if item.provider == provider_name)
            if provider_name not in eligible:
                stage_reports.append({
                    "provider": provider_name, "mode": "end_to_end_canary",
                    "status": "SKIPPED_PREFLIGHT_FAILED", "records": 0,
                })
                continue
            if halted:
                stage_reports.append({
                    "provider": provider_name, "mode": "end_to_end_canary",
                    "status": "SKIPPED_ERROR_THRESHOLD", "records": 0,
                })
                continue
            canary, canary_report = self._run_stage(
                model, "end_to_end", resume, canary=True,
            )
            canary_records.extend(canary)
            stage_reports.append(canary_report)
            canary_passed = (
                canary_report["records"] == len(self.config.methods["end_to_end"])
                and canary_report["structured_valid_records"] == canary_report["records"]
                and canary_report["infrastructure_errors"] == 0
                and canary_report["tool_call_interface_verified"]
            )
            if not canary_passed:
                canary_report["status"] = "FAILED_CANARY_GATE"
                halted = True
                continue
            stage_records, report = self._run_stage(model, "end_to_end", resume)
            records.extend(stage_records)
            stage_reports.append(report)
            if report["infrastructure_error_rate"] > 0.20:
                halted = True

        prompt_freeze = self._prompt_freeze_report()
        public_records = [strip_evaluator_fields(item) for item in records]
        summary = {
            "experiment_id": self.config.name,
            "experiment_stage": "PILOT", "publication_status": "DEVELOPMENT_ONLY",
            "run_scope": "end_to_end_only",
            "component_records_rerun": 0,
            "prior_component_attempt": "pilot_v2",
            "real_model_records": len(public_records), "planned_real_model_records": 120,
            "canary_records": len(canary_records), "symbolic_upper_bound_records": 0,
            "stage_reports": stage_reports,
            "metrics": aggregate_metrics(records),
            "metrics_by_method_mode": by_method_and_mode(records),
            "cost": self.costs.snapshot(), "prompt_freeze": prompt_freeze,
            "ground_truth_leakage_count": 0, "api_key_leakage_count": 0,
            "unsafe_real_world_actions": 0,
            "full_real_model_experiment_status": "NOT RUN",
            "halted_on_infrastructure_error_rate": halted,
        }
        for model in self.config.models:
            assert_secret_absent(summary, os.getenv(model.api_key_env or ""))
        self._atomic(self.output / "summary.json", summary)
        self._write_forecast(records, prompt_freeze)
        return summary

    def _run_stage(self, model, mode: str, resume: bool, canary: bool = False):
        methods = self.config.methods[mode]
        seed = self.config.seeds[0]
        view = self.config.phase9_view(model, mode, methods, seed)
        directory = (
            self.output / "canary" / f"{model.provider}-{model.model_id}"
            if canary else self.output / mode / f"{model.provider}-{model.model_id}"
        )
        checkpoints = CheckpointStore(directory)
        cache = LLMCache(self.output / ".cache" / model.provider / mode)
        limiter = RequestRateLimiter(self.config.execution.get("rate_limit_per_second"))
        api_key = os.getenv(model.api_key_env or "")
        if not api_key:
            raise ValueError(f"{model.api_key_env}: missing")

        def provider_factory(_outputs):
            if self.provider_builder is not None:
                return self.provider_builder(view.model, _outputs)
            return OpenAICompatibleProvider(
                view.model, api_key, rate_limiter=limiter, cost_controller=self.costs,
            )

        runner = ExperimentScenarioRunner(view, cache, self.prompts, self.schemas, provider_factory)
        jobs = []
        resumed_records = []
        selected_families = self.families[:1] if canary else self.families
        for family in selected_families:
            selected_scenarios = family["scenarios"][:1] if canary else family["scenarios"]
            for scenario in selected_scenarios:
                for method in methods:
                    record_id = self._record_id(model.provider, model.model_id, mode, scenario["scenario_id"], method, seed)
                    if resume and checkpoints.has(record_id):
                        record = checkpoints.read(record_id)
                        record["_expected"] = _expected(scenario)
                        resumed_records.append(record)
                    else:
                        jobs.append((family, scenario, method, record_id))

        def execute(job):
            family, scenario, method, record_id = job
            try:
                internal = runner.run(family, scenario, method, mode, 0)
                public = self._decorate(internal, model, seed, family)
            except Exception as exc:
                internal, public = self._error_record(family, scenario, method, mode, model, seed, exc, runner)
            for secret_model in self.config.models:
                assert_secret_absent(public, os.getenv(secret_model.api_key_env or ""))
            checkpoints.write(record_id, public)
            return {**public, "_expected": internal["_expected"]}

        created = []
        workers = min(model.concurrency, max(1, len(jobs)))
        if jobs:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                created.extend(executor.map(execute, jobs))
        all_records = resumed_records + created
        errors = sum(item.get("error_category") in INFRASTRUCTURE_ERRORS for item in all_records)
        report = {
            "provider": model.provider, "configured_model": model.model_id,
            "mode": f"{mode}_canary" if canary else mode,
            "status": "COMPLETED", "records": len(all_records), "created": len(created),
            "resumed": len(resumed_records), "infrastructure_errors": errors,
            "infrastructure_error_rate": errors / len(all_records) if all_records else 0.0,
        }
        if canary and mode == "end_to_end":
            report.update({
                "structured_valid_records": sum(
                    item.get("error_category") not in INFRASTRUCTURE_ERRORS
                    for item in all_records
                ),
                "parsed_tool_call_records": sum(
                    int(item.get("tool_calls", 0)) > 0 for item in all_records
                ),
                "tool_call_interface_verified": any(
                    int(item.get("tool_calls", 0)) > 0 for item in all_records
                ),
            })
        checkpoints.write_summary(report)
        return all_records, report

    def _decorate(self, internal, model, seed, family):
        public = strip_evaluator_fields(internal)
        actual_models = internal.get("_actual_models", [])
        public.update({
            "provider": model.provider,
            "configured_model": model.model_id,
            "actual_model": actual_models[-1] if actual_models else model.model_id,
            "seed": seed,
            "experiment_stage": "PILOT",
            "publication_status": "DEVELOPMENT_ONLY",
            "api_calls": int(internal.get("_provider_attempts", 0)),
            "estimated_cost_cny": self.catalog.estimate_cny(model, public["input_tokens"], public["output_tokens"]),
            "failure_analysis": _failure_analysis(public, internal["_expected"]),
            "public_family_id": f"family-{hashlib.sha256(family['matched_case_id'].encode()).hexdigest()[:12]}",
        })
        return public

    def _error_record(self, family, scenario, method, mode, model, seed, exc, runner):
        error = _error_category(exc)
        expected = _expected(scenario)
        internal = runner._record(
            _public_id(scenario["scenario_id"]), mode=mode, method=method, repetition=0,
            success=False, predicted=None, localization=None, patch_proposed=False,
            patch_accepted=False, immediate=None, transfer=None, termination=error,
            error=error,
            prompt_hash=(runner._component_prompt_hash(method) if mode == "component" else runner._prompt_hash(method)),
            expected=expected,
        )
        internal["_actual_models"], internal["_provider_attempts"] = [], 0
        public = self._decorate(internal, model, seed, family)
        public["failure_analysis"] = ["provider_failure" if error.startswith("PROVIDER") or error == "RATE_LIMITED" else "task_runner_internal_error"]
        public["sanitized_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return internal, public

    def _run_symbolic(self):
        directory = self.output / "symbolic_upper_bound"
        checkpoints = CheckpointStore(directory)
        model = self.config.models[0]
        view = self.config.phase9_view(model, "component", ("oracle_symbolic_upper_bound",), self.config.seeds[0])
        runner = ExperimentScenarioRunner(view, None, self.prompts, self.schemas, lambda outputs: MockProvider(outputs))
        records = []
        for family in self.families:
            for scenario in family["scenarios"]:
                internal = runner.run(family, scenario, "oracle_symbolic_upper_bound", "component", 0)
                public = strip_evaluator_fields(internal)
                public.update({
                    "provider": "symbolic", "configured_model": "oracle_symbolic_upper_bound",
                    "actual_model": "oracle_symbolic_upper_bound", "seed": self.config.seeds[0],
                    "experiment_stage": "PILOT", "publication_status": "DEVELOPMENT_ONLY",
                    "api_calls": 0, "estimated_cost_cny": 0.0, "failure_analysis": [],
                    "public_family_id": f"family-{hashlib.sha256(family['matched_case_id'].encode()).hexdigest()[:12]}",
                })
                checkpoints.write(hashlib.sha256(public["public_scenario_id"].encode()).hexdigest()[:24], public)
                records.append(public)
        checkpoints.write_summary({"records": len(records), "api_calls": 0, "estimated_cost_cny": 0.0})
        return records

    def _preflight_summary(self):
        path = self.output.parent / "preflight" / "summary.json"
        if not path.exists():
            raise PermissionError("successful Phase 10 preflight is required before Pilot")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self):
        pricing_path = PROJECT_ROOT / "benchmark/pricing/phase10_pricing_v1.json"
        manifest = {
            "experiment_id": self.config.name, "experiment_stage": "PILOT",
            "publication_status": "DEVELOPMENT_ONLY", "config_hash": self.config.config_hash,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "dirty_working_tree": bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()),
            "models": [model.public_dict() for model in self.config.models],
            "prompt_hashes": self.prompt_hashes, "pricing_snapshot_hash": file_hash(pricing_path),
            "pricing_snapshot_id": self.catalog.document["snapshot_id"],
            "benchmark_hashes": {path: file_hash(PROJECT_ROOT / path) for path in PROTECTED_PATHS},
            "families": list(self.config.families), "seeds": list(self.config.seeds),
            "methods": {key: list(value) for key, value in self.config.methods.items()},
            "budget": self.config.budget.__dict__, "full_real_model_experiment_status": "NOT RUN",
        }
        if self.config.raw.get("experiment", {}).get("action_prompt_version") == "base_tool_agent_v3":
            manifest["policy_capabilities"] = {
                method: list(PolicyCapabilities.for_method(method).allowed_actions)
                for method in self.config.methods.get("end_to_end", ())
            }
            manifest["behavior_affecting_prompt_format_change"] = True
        for model in self.config.models:
            assert_secret_absent(manifest, os.getenv(model.api_key_env or ""))
        path = self.output / "manifest.json"
        if path.exists() and json.loads(path.read_text())["config_hash"] != self.config.config_hash:
            raise ValueError("Pilot manifest config mismatch")
        if not path.exists():
            self._atomic(path, manifest)

    def _prompt_freeze_report(self):
        before_path = self.output / "prompt_freeze_before.json"
        before = json.loads(before_path.read_text())
        v1_hashes = self._v1_component_hashes()
        v2_hashes = {}
        for method in self.config.methods["component"]:
            view = self.config.phase9_view(self.config.models[0], "component", (method,), self.config.seeds[0])
            runner = ExperimentScenarioRunner(view, None, self.prompts, self.schemas)
            v2_hashes[method] = runner._component_prompt_hash(method)
        report = {
            "original_prompt_hashes": before["prompt_hashes"], "current_prompt_hashes": self.prompt_hashes,
            "v1_component_prompt_hashes": v1_hashes,
            "v2_component_prompt_hashes": v2_hashes,
            "modified_after_pilot_start": True,
            "behavior_affecting_change": True,
            "change_reason": "The strict-JSON component prompt now includes the complete LLMAttribution schema and a neutral structural example.",
            "pilot_families_are_development_set": True,
            "development_families": ["M01", "M06", "M11", "M16"],
            "recommended_main_config": "phase10_main_heldout48.yaml",
        }
        if self.config.raw.get("experiment", {}).get("run_scope") == "end_to_end_only":
            old_action_hashes = self._prior_action_hashes()
            new_action_hashes = {}
            for method in self.config.methods["end_to_end"]:
                view = self.config.phase9_view(
                    self.config.models[0], "end_to_end", (method,), self.config.seeds[0],
                )
                runner = ExperimentScenarioRunner(view, None, self.prompts, self.schemas)
                new_action_hashes[method] = runner._prompt_hash(method)
            report.update({
                "v1_agent_action_prompt_hashes": old_action_hashes,
                "v2_agent_action_prompt_hashes": new_action_hashes,
                "component_prompt_v2_modified": False,
                "component_records_rerun": 0,
                "change_reason": "The strict end-to-end action prompt now includes the complete action-specific AgentAction schema and neutral field guidance.",
            })
        self._atomic(self.output / "prompt_freeze_report.json", report)
        return report

    def _prior_action_hashes(self):
        root = self.output.parent / "pilot_v2" / "end_to_end"
        values: dict[str, str] = {}
        for path in root.glob("*/records/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            values.setdefault(record["method"], record["prompt_hash"])
        return values

    def _v1_component_hashes(self):
        root = self.output.parent / "pilot" / "component"
        values: dict[str, str] = {}
        for path in root.glob("*/records/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            values.setdefault(record["method"], record["prompt_hash"])
        return values

    def _ensure_prompt_freeze_before(self):
        path = self.output / "prompt_freeze_before.json"
        if not path.exists():
            self._atomic(path, {"prompt_hashes": self.prompt_hashes, "frozen_before_first_pilot_call": True})

    def _write_forecast(self, records, freeze):
        real = [item for item in records if item.get("provider") in {"deepseek", "dashscope"}]
        by_provider = {}
        for model in self.config.models:
            rows = [item for item in real if item["provider"] == model.provider]
            average = sum(item["estimated_cost_cny"] for item in rows) / len(rows) if rows else 0.0
            by_provider[model.provider] = {"pilot_records": len(rows), "average_cost_per_record_cny": average}
        forecast = {
            "estimate_only": True, "actual_bill_may_differ": True,
            "main_all60_records": 1800, "main_heldout48_records": 1440,
            "full_soft_limit_cny": self.config.budget.full_soft_limit_cny,
            "full_hard_limit_cny": self.config.budget.full_hard_limit_cny,
            "by_provider": by_provider,
            "recommended_config": freeze["recommended_main_config"],
        }
        average_all = sum(item["estimated_cost_cny"] for item in real) / len(real) if real else 0.0
        forecast["all60_estimated_cny"] = average_all * 1800
        forecast["heldout48_estimated_cny"] = average_all * 1440
        attempt = str(self.config.raw.get("experiment", {}).get("pilot_attempt", "v1"))
        self._atomic(self.output.parent / "cost" / f"full_run_forecast_{attempt}.json", forecast)

    def _record_id(self, provider, model, mode, scenario, method, seed):
        material = f"{self.config.config_hash}|{provider}|{model}|{mode}|{scenario}|{method}|0|{seed}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    @staticmethod
    def _atomic(path: Path, value: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def _public_id(scenario_id: str) -> str:
    return f"public-{hashlib.sha256(scenario_id.encode()).hexdigest()[:12]}"


def _expected(scenario):
    return {"agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"}[
        scenario["evaluator_metadata"]["ground_truth_label"]
    ]


def _error_category(exc):
    name = type(exc).__name__
    if name == "ProviderTimeout": return "PROVIDER_TIMEOUT"
    if name == "RateLimited": return "RATE_LIMITED"
    if name == "ProviderError": return "PROVIDER_ERROR"
    if name == "InvalidStructuredOutput": return "INVALID_STRUCTURED_OUTPUT"
    if isinstance(exc, CostHardLimit): return "BUDGET_EXHAUSTED"
    return "INTERNAL_ERROR"


def _failure_analysis(record, expected):
    values = []
    error = record.get("error_category")
    if error in {"UNKNOWN_TOOL", "INVALID_ARGUMENTS", "INVALID_STRUCTURED_OUTPUT", "BUDGET_EXHAUSTED"}:
        values.append(error.lower())
    if error in {"PROVIDER_ERROR", "PROVIDER_TIMEOUT", "RATE_LIMITED"}:
        values.append("provider_failure")
    if record.get("predicted_class") not in {None, expected}:
        values.append(f"misclassified_{expected.lower()}")
    if record.get("target_tool_correct") is False: values.append("wrong_tool_localization")
    if record.get("drift_category_correct") is False: values.append("wrong_drift_category")
    if record.get("exact_location_correct") is False: values.append("wrong_location")
    if not record.get("final_task_success") and not values: values.append("task_planning_failure")
    if record.get("future_transfer") is False: values.append("future_transfer_failure")
    return values
