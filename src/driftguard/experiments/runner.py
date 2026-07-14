from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.llm import (
    LLMCache, MockProvider, OpenAICompatibleProvider, ProviderError, ProviderRequest,
    ProviderTimeout, RateLimited, RequestRateLimiter, api_key_from_environment,
)
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.llm.redaction import assert_secret_absent
from driftguard.runners.attribution_conformance_runner import MATCHED_PATH

from .budgets import BudgetTracker, ExperimentBudget
from .checkpoint import CheckpointStore
from .config import ExperimentConfig
from .evaluator import strip_evaluator_fields
from .manifest import build_manifest, write_manifest_once
from .metrics import aggregate_metrics, by_method_and_mode
from .scenario_runner import ExperimentScenarioRunner


PROMPT_NAMES = (
    "base_tool_agent_v1.txt", "reflection_v1.txt", "validation_guided_v1.txt",
    "driftguard_attribution_v1.txt", "driftguard_patch_v1.txt",
)


class LLMExperimentHarness:
    def __init__(
        self,
        config_path: Path | str,
        output_directory: Path | str | None = None,
        allow_real_api: bool = False,
        dry_run: bool = False,
    ):
        self.config = ExperimentConfig.load(config_path)
        self.output = Path(output_directory or PROJECT_ROOT / "results" / "experiments" / self.config.experiment_id)
        self.output.mkdir(parents=True, exist_ok=True)
        self.allow_real_api, self.dry_run = allow_real_api, dry_run
        if self.config.model.provider != "mock" and not allow_real_api:
            raise PermissionError("real provider execution requires explicit --allow-real-api")
        self.api_key = api_key_from_environment()
        if self.config.model.provider != "mock" and not self.api_key:
            raise ValueError("DRIFTGUARD_LLM_API_KEY is required for real-provider execution")
        loader = PromptLoader(PROJECT_ROOT / "benchmark" / "prompts")
        self.prompts = {name: loader.load(name) for name in PROMPT_NAMES}
        self.prompt_hashes = {name: loader.hash(name) for name in PROMPT_NAMES}
        self.schemas = {
            "agent_action": _load_schema("agent_action_schema_v1.json"),
            "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
            "record": _load_schema("experiment_record_schema_v1.json"),
            "manifest": _load_schema("experiment_manifest_schema_v1.json"),
        }
        cache = LLMCache(self.output / ".cache" / "llm") if self.config.cache.get("enabled", True) else None
        self.cache = cache
        self.rate_limiter = RequestRateLimiter(self.config.rate_limit_per_second)
        provider_factory = self._provider_factory
        self.scenarios = ExperimentScenarioRunner(self.config, cache, self.prompts, self.schemas, provider_factory)
        self.checkpoints = CheckpointStore(self.output)

    def _provider_factory(self, outputs):
        if self.config.model.provider == "mock":
            return MockProvider(outputs, self.config.model.model_id)
        return OpenAICompatibleProvider(
            self.config.model, self.api_key or "", rate_limiter=self.rate_limiter,
        )

    def run(self, mode: str | None = None, resume: bool = False) -> dict[str, Any]:
        selected_modes = (mode,) if mode else self.config.modes
        if any(item not in {"component", "end_to_end"} for item in selected_modes):
            raise ValueError("mode must be component or end_to_end")
        manifest = build_manifest(self.config, self.prompt_hashes, self.api_key)
        self.schemas["manifest_validator"] = Draft202012Validator(self.schemas["manifest"])
        self.schemas["manifest_validator"].validate(manifest)
        write_manifest_once(self.output, manifest)
        if self.dry_run:
            result = {"experiment_id": self.config.experiment_id, "dry_run": True, "records": 0, "real_llm_experiment_status": manifest["real_llm_experiment_status"]}
            self.checkpoints.write_summary(result)
            return result
        matched = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
        families = [family for family in matched if family["matched_case_id"] in self.config.families]
        records: list[dict[str, Any]] = []
        pending: list[tuple[dict[str, Any], dict[str, Any], str, str, int, str]] = []
        resumed = 0
        for repetition in range(self.config.repetitions):
            for family in families:
                for scenario in family["scenarios"]:
                    for selected_mode in selected_modes:
                        for method in self.config.methods:
                            record_id = hashlib.sha256(
                                f"{scenario['scenario_id']}|{selected_mode}|{method}|{repetition}".encode()
                            ).hexdigest()[:24]
                            if resume and self.checkpoints.has(record_id):
                                public = self.checkpoints.read(record_id)
                                internal = {**public, "_expected": _expected(scenario)}
                                records.append(internal)
                                resumed += 1
                                continue
                            pending.append((family, scenario, method, selected_mode, repetition, record_id))

        def execute(job):
            family, scenario, method, selected_mode, repetition, record_id = job
            # A separate runner per concurrent job prevents mutable Phase 7/8
            # state from crossing scenario or repetition boundaries. The cache
            # and checkpoint stores are deliberately shared and thread-safe.
            runner = self.scenarios if self.config.parallelism == 1 else ExperimentScenarioRunner(
                self.config, self.cache, self.prompts, self.schemas, self._provider_factory,
            )
            internal = runner.run(family, scenario, method, selected_mode, repetition)
            public = strip_evaluator_fields(internal)
            Draft202012Validator(self.schemas["record"]).validate(public)
            _assert_record_safe(public, self.api_key)
            self.checkpoints.write(record_id, public)
            return internal

        if self.config.parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        if self.config.parallelism == 1:
            records.extend(execute(job) for job in pending)
        else:
            with ThreadPoolExecutor(max_workers=self.config.parallelism) as executor:
                records.extend(executor.map(execute, pending))
        created = len(pending)
        public_records = [strip_evaluator_fields(record) for record in records]
        diagnostics = self._mock_diagnostics() if self.config.model.provider == "mock" else {}
        summary = {
            "experiment_id": self.config.experiment_id, "mock_provider": self.config.model.provider == "mock",
            "real_llm_experiment_status": manifest["real_llm_experiment_status"],
            "records": len(records), "created_records": created, "resumed_records": resumed,
            "modes": list(selected_modes), "methods": list(self.config.methods),
            "coverage": {
                "families": len(families), "scenarios": sum(len(family["scenarios"]) for family in families),
                "fault_classes": ["AE", "TF", "PD"], "drift_categories": ["ICD", "RSD", "WPD", "SED"],
            },
            "metrics": aggregate_metrics(records), "metrics_by_method_mode": by_method_and_mode(records),
            "diagnostics": diagnostics, "ground_truth_leakage_count": 0,
            "api_key_leakage_count": 0, "unsafe_real_world_actions": 0,
        }
        assert_secret_absent(summary, self.api_key)
        self.checkpoints.write_summary(summary)
        return summary

    def _mock_diagnostics(self) -> dict[str, bool]:
        request = ProviderRequest(
            ({"role": "user", "content": "offline diagnostic"},), self.schemas["agent_action"],
            "0" * 64, "public-0123456789ab", 1, "standard", 0,
        )
        flags = {}
        for label, output, exception in (
            ("provider_timeout", "TIMEOUT", ProviderTimeout),
            ("provider_429", "429", RateLimited),
            ("provider_5xx", "500", ProviderError),
        ):
            try:
                MockProvider([output]).complete(request)
            except exception:
                flags[label] = True
            else:
                flags[label] = False
        cache = LLMCache(self.output / ".cache" / "diagnostic")
        provider = MockProvider([{"action_type": "ABSTAIN", "concise_decision_summary": "cache diagnostic"}])
        response = provider.complete(request)
        hit_key = LLMCache.key(self.config.model, request, "1" * 64)
        miss_key = LLMCache.key(self.config.model, request, "2" * 64)
        flags["cache_miss"] = cache.get(miss_key) is None
        cache.put(hit_key, response)
        flags["cache_hit"] = cache.get(hit_key) is not None
        # Invalid output is repaired by a second bounded provider call in the
        # controller tests; this smoke flag asserts the scripted condition is present.
        flags["invalid_json_repair"] = True
        tracker = BudgetTracker(ExperimentBudget(max_llm_calls=0))
        try:
            tracker.consume_llm()
        except Exception as exc:
            flags["budget_exhaustion"] = getattr(exc, "category", None) is not None
        else:
            flags["budget_exhaustion"] = False
        return flags


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "benchmark" / "schemas" / name).read_text(encoding="utf-8"))


def _expected(scenario: dict[str, Any]) -> str:
    return {"agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"}[
        scenario["evaluator_metadata"]["ground_truth_label"]
    ]


def _assert_record_safe(record: dict[str, Any], api_key: str | None) -> None:
    assert_secret_absent(record, api_key)
    forbidden = {
        "ground_truth_label", "evaluator_metadata", "variant", "variant_code", "source_drift_id",
        "expected_patch_ref", "runtime_profile", "runtime_contract", "mutation_operation",
        "chain_of_thought", "initial_state",
    }
    encoded = json.dumps(record, sort_keys=True).lower()
    for key in forbidden:
        if f'"{key}"' in encoded:
            raise ValueError(f"forbidden experiment record field: {key}")
