from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.budgets import BudgetExhausted
from driftguard.llm import (
    InvalidStructuredOutput, LLMCache, ModelConfig, OpenAICompatibleProvider,
    ProviderCapabilityAdapter, ProviderError, ProviderRequest, RequestRateLimiter, model_api_key,
    parse_structured_output,
)
from driftguard.llm.models import ProviderResponse
from driftguard.llm.redaction import assert_secret_absent
from driftguard.phase10.pricing import PricingCatalog
from driftguard.runners.focused_live_healing_runner import (
    AtomicFocusedLedger, LEDGER, _ledger_dominates, _ledger_lock,
    _public_ledger_snapshot, _sha256_bytes, _validated_ledger,
)

from .protocol import (
    BenchmarkErrorClass, CanonicalToolRegistry, EVIDENCE_VIEWS, FAMILIES,
    VARIANTS, EvidenceViewBuilder, OutputTruncated, TrueEvidenceLeakage,
    assert_benchmark_visible, balanced_evidence_view, classify_error,
    expected_for_evaluator, normalize_prediction,
)


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/specdriftbench_component_canary_v1.yaml"
PROMPT_PATH = PROJECT_ROOT / "benchmark/prompts/specdriftbench_component_v1.txt"
OUTPUT_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/benchmark_attribution_schema_v1.json"
RECORD_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/specdriftbench_component_record_schema_v2.json"
MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/specdriftbench_component_manifest_schema_v2.json"
VIEW_DEFINITION_PATH = PROJECT_ROOT / "benchmark/evidence_views/specdriftbench_evidence_views_v1.json"
PRICING_PATH = PROJECT_ROOT / "benchmark/pricing/specdriftbench_pricing_v1.json"
OLD_ATTEMPT = PROJECT_ROOT / "results/experiments/phase10/driftguard_focused_canary/real_attempts/phase10a-v3-20260717-eb2efa85682d"
CONFIRM_36 = "RUN-EXACTLY-36"
CONFIRM_PREFLIGHT = "RUN-KIMI-PREFLIGHT-1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _source_snapshot() -> str:
    paths = (
        "src/driftguard/specdriftbench/protocol.py",
        "src/driftguard/specdriftbench/runner.py",
        "src/driftguard/llm/provider.py",
        "src/driftguard/llm/provider_capabilities.py",
        "src/driftguard/runners/attribution_conformance_runner.py",
        "benchmark/prompts/specdriftbench_component_v1.txt",
        "benchmark/schemas/benchmark_attribution_schema_v1.json",
        "benchmark/evidence_views/specdriftbench_evidence_views_v1.json",
    )
    hashes = {path: _sha256_bytes(PROJECT_ROOT / path) for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CanaryPlan:
    ordinal: int
    provider: str
    model: str
    family: str
    variant: str
    evidence_view: str

    @property
    def record_id(self) -> str:
        material = f"specdriftbench-v1|{self.provider}|{self.model}|{self.family}|{self.variant}|{self.evidence_view}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal, "record_id": self.record_id,
            "provider": self.provider, "model": self.model,
            "family": self.family, "variant": self.variant,
            "evidence_view": self.evidence_view,
        }


@dataclass(frozen=True)
class SpecDriftBenchConfig:
    path: Path
    raw: dict[str, Any]
    models: tuple[ModelConfig, ...]
    plan: tuple[CanaryPlan, ...]
    config_hash: str

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "SpecDriftBenchConfig":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        models = tuple(ModelConfig.from_mapping(item) for item in raw["models"])
        providers = [(item.provider, item.model_id) for item in models]
        if providers != [
            ("deepseek", "deepseek-v4-flash"),
            ("dashscope", "qwen3.7-plus"),
            ("moonshot", "kimi-k2.6"),
        ]:
            raise ValueError("SpecDriftBench Provider order/configuration changed")
        experiment = raw["experiment"]
        if tuple(experiment["development_families"]) != FAMILIES or tuple(experiment["variants"]) != VARIANTS:
            raise ValueError("development family/variant selection changed")
        if experiment["heldout48_allowed"] or experiment["phase10b_allowed"] or experiment["end_to_end_allowed"]:
            raise PermissionError("SpecDriftBench Canary cannot authorize heldout, Phase 10B, or end-to-end")
        plan: list[CanaryPlan] = []
        ordinal = 0
        for model in models:
            for family in FAMILIES:
                for variant in VARIANTS:
                    ordinal += 1
                    plan.append(CanaryPlan(
                        ordinal, model.provider, model.model_id, family, variant,
                        balanced_evidence_view(family, variant),
                    ))
        if len(plan) != experiment["planned_records"] or len(plan) != 36:
            raise ValueError("SpecDriftBench plan must contain exactly 36 records")
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        return cls(source, raw, models, tuple(plan), hashlib.sha256(encoded.encode()).hexdigest())

    def model(self, provider: str) -> ModelConfig:
        return next(item for item in self.models if item.provider == provider)

    @property
    def authorized(self) -> bool:
        return bool(self.raw["experiment"].get("run_authorized"))


class _LedgerConfigAdapter:
    def __init__(self, config: SpecDriftBenchConfig):
        self.config_hash = config.config_hash
        budget = config.raw["budget"]
        self.raw = {
            "execution": {
                "attempt_soft_increment_cny": budget["attempt_soft_increment_cny"],
                "attempt_hard_increment_cny": budget["attempt_hard_increment_cny"],
                "total_hard_limit_cny": budget["total_hard_limit_cny"],
            },
            "budget": {
                "full_hard_limit_cny": budget["total_hard_limit_cny"],
                "max_input_tokens_per_record": budget["max_input_tokens_per_record"],
                "max_output_tokens_per_record": budget["max_output_tokens_per_record"],
            },
        }


def capture_ledger_baseline(path: Path = LEDGER) -> tuple[dict[str, Any], str]:
    with _ledger_lock(path):
        current, digest = _validated_ledger(path)
        baseline = _public_ledger_snapshot(current)
        if path.resolve() == LEDGER.resolve():
            tracked = subprocess.run(
                ["git", "show", "HEAD:results/experiments/phase10/cost/ledger.json"],
                cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
            )
            if tracked.returncode == 0:
                floor = _public_ledger_snapshot(json.loads(tracked.stdout))
                if not _ledger_dominates(baseline, floor):
                    raise ValueError("formal Ledger rollback detected")
        if baseline["spent_cny"] > 50:
            raise ValueError("formal Ledger cumulative hard limit exceeded")
        return baseline, digest


class SpecDriftBenchRunner:
    def __init__(
        self, config_path: Path | str = CONFIG_PATH, *, attempt_id: str,
        mode: str, allow_real_api: bool = False, confirmation: str | None = None,
        output_root: Path | None = None, ledger_path: Path = LEDGER,
    ) -> None:
        if mode not in {"real", "replay"}:
            raise ValueError("mode must be real or replay")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id):
            raise ValueError("unsafe attempt ID")
        self.config = SpecDriftBenchConfig.load(config_path)
        self.attempt_id, self.mode = attempt_id, mode
        self.allow_real_api, self.confirmation = allow_real_api, confirmation
        self.ledger_path = Path(ledger_path)
        base = PROJECT_ROOT / "results/experiments/phase10/specdriftbench_component_canary/attempts" / attempt_id
        self.root = Path(output_root or base)
        self.results = self.root / ("results" if mode == "real" else "replay")
        self.builder = EvidenceViewBuilder()
        self.registry = CanonicalToolRegistry.from_displayed_spec()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.schema = json.loads(OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.record_validator = Draft202012Validator(json.loads(RECORD_SCHEMA_PATH.read_text()))
        self.manifest_validator = Draft202012Validator(json.loads(MANIFEST_SCHEMA_PATH.read_text()))
        self.pricing = PricingCatalog(json.loads(PRICING_PATH.read_text(encoding="utf-8")))
        self.network_calls = 0
        self.provider_instances = 0
        self.ledger_before, self.ledger_before_hash = self._baseline_from_manifest_or_current()
        self.invocation_before, self.invocation_before_hash = capture_ledger_baseline(self.ledger_path)
        settled = self._settled_attempt_cost()
        self.costs = (
            AtomicFocusedLedger(
                self.ledger_path, _LedgerConfigAdapter(self.config), attempt_id,
                self.ledger_before, settled, pricing_catalog=self.pricing,
            ) if mode == "real" else None
        )
        if mode == "real":
            if not (self.config.authorized and allow_real_api and confirmation == CONFIRM_36):
                raise PermissionError("real SpecDriftBench requires authorized config, --allow-real-api and RUN-EXACTLY-36")
            self._check_credentials()
        self._write_or_validate_manifest()

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        completed: list[dict[str, Any]] = []
        globally_stopped = None
        for provider in ("deepseek", "dashscope", "moonshot"):
            provider_records: list[dict[str, Any]] = []
            model = self.config.model(provider)
            cache = self._cache(provider)
            boundary = self._provider(model) if self.mode == "real" else None
            for plan in [item for item in self.config.plan if item.provider == provider]:
                checkpoint = self.results / "checkpoint" / provider / f"{plan.record_id}.json"
                if checkpoint.exists():
                    if not resume:
                        raise FileExistsError(f"checkpoint exists for {plan.record_id}; use --resume")
                    record = json.loads(checkpoint.read_text(encoding="utf-8"))
                else:
                    record = self._run_record(plan, model, cache, boundary)
                    _atomic_json(checkpoint, record)
                if self.mode == "replay":
                    original_path = self.root / "results/records" / f"{plan.record_id}.json"
                    if not original_path.exists():
                        raise CacheReplayMiss(f"original record missing for {plan.record_id}")
                    original = json.loads(original_path.read_text(encoding="utf-8"))
                    for field in ("raw_response_sha256", "parsed_result", "normalized_prediction"):
                        if record.get(field) != original.get(field):
                            raise RuntimeError(f"replay mismatch for {plan.record_id}: {field}")
                self.record_validator.validate(record)
                _atomic_json(self.results / "records" / f"{plan.record_id}.json", record)
                completed.append(record); provider_records.append(record)
                self._write_summary(completed, "INCOMPLETE")
                if record["error_class"] == BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE.value:
                    globally_stopped = "TRUE_EVIDENCE_LEAKAGE"
                    break
                if self._is_infrastructure(record) and sum(self._is_infrastructure(item) for item in provider_records) >= 2:
                    break
            gate = {
                "provider": provider, "completed": len(provider_records),
                "infrastructure_errors": sum(self._is_infrastructure(item) for item in provider_records),
                "continued_to_next_provider": globally_stopped is None,
            }
            _atomic_json(self.results / "provider_gates" / f"{provider}.json", gate)
            if globally_stopped:
                break
        status = "COMPLETE" if len(completed) == 36 else "STOPPED"
        summary = self._write_summary(completed, status)
        if globally_stopped:
            summary["global_stop_reason"] = globally_stopped
            _atomic_json(self.results / "summary.json", summary)
        if self.mode == "replay":
            after, after_hash = capture_ledger_baseline(self.ledger_path)
            if after_hash != self.invocation_before_hash or after != self.invocation_before:
                raise RuntimeError("formal Ledger changed during replay")
        return {"summary": summary, "records": completed}

    def _run_record(
        self, plan: CanaryPlan, model: ModelConfig, cache: LLMCache,
        provider: OpenAICompatibleProvider | None,
    ) -> dict[str, Any]:
        record_ledger_before = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        evidence = self.builder.build(plan.family, plan.variant, plan.evidence_view)
        expected = expected_for_evaluator(self.builder, plan.family, plan.variant)
        assert_benchmark_visible(evidence)
        output_schema_text = json.dumps(self.schema, indent=2, sort_keys=True)
        system = self.prompt.replace("{{OUTPUT_SCHEMA}}", output_schema_text)
        messages: tuple[dict[str, str], ...] = (
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, default=list)},
        )
        prompt_hash = hashlib.sha256(system.encode()).hexdigest()
        schema_hash = _sha256_bytes(OUTPUT_SCHEMA_PATH)
        responses: list[ProviderResponse] = []
        parsed = normalized = None
        error: BaseException | None = None
        raw_ref = raw_sha = None
        repairs = 0
        for repair_index in range(2):
            request = ProviderRequest(
                messages, self.schema, prompt_hash, evidence["public_scenario_id"],
                5, "component_attribution", 0, self.config.raw["execution"]["seed"],
                f"specdriftbench_{plan.evidence_view.lower()}", self.config.config_hash,
            )
            key = LLMCache.key(model, request, schema_hash)
            response = cache.get(key)
            try:
                if response is None:
                    if self.mode == "replay":
                        raise CacheReplayMiss(f"cache miss for {plan.provider}/{plan.family}-{plan.variant}/{plan.evidence_view}")
                    if provider is None:
                        raise RuntimeError("real Provider is unavailable")
                    response = provider.complete(request)
                    self.network_calls += 1
                    cache.put(key, response, model_api_key(model))
                responses.append(response)
                raw_ref = f"{plan.provider}/{key}.json"
                raw_sha = hashlib.sha256(response.raw_text.encode()).hexdigest()
                assert_benchmark_visible(response.raw_text)
                if response.finish_reason == "length":
                    raise OutputTruncated(response.finish_reason, len(response.raw_text))
                parsed = parse_structured_output(response.raw_text, self.schema)
                assert_benchmark_visible(parsed)
                normalized = normalize_prediction(parsed, self.registry)
                if not response.cached:
                    response = replace(response, parsed_output=parsed)
                    cache.put(key, response, model_api_key(model))
                    responses[-1] = response
                break
            except InvalidStructuredOutput as exc:
                error = exc
                if repair_index == 1:
                    break
                repairs = 1
                messages = messages + (
                    {"role": "assistant", "content": response.raw_text if response else "{}"},
                    {"role": "user", "content": "Schema validation error: " + str(exc) + "\nReturn one corrected JSON object matching the unchanged schema."},
                )
            except BaseException as exc:
                error = exc
                break
        if normalized is not None:
            error = None
        error_class = BenchmarkErrorClass.NONE if error is None else classify_error(error)
        if self.mode == "replay" and isinstance(error, CacheReplayMiss):
            raise error
        usage = {
            "input": sum(item.input_tokens for item in responses),
            "output": sum(item.output_tokens for item in responses),
            "latency": sum(item.latency_ms for item in responses),
            "attempts": sum(item.provider_attempts for item in responses),
            "reasoning": sum(item.reasoning_tokens for item in responses),
        }
        after = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        evaluation = self._evaluate(normalized, expected)
        record = {
            "schema_version": "specdriftbench-component-record-v2",
            "record_id": plan.record_id, "attempt_id": self.attempt_id,
            "provider": plan.provider, "model": plan.model,
            "provider_profile_id": ProviderCapabilityAdapter.for_model(model).profile_id,
            "max_tokens": model.max_output_tokens,
            "family": plan.family, "variant": plan.variant,
            "evidence_view": plan.evidence_view,
            "public_scenario_id": evidence["public_scenario_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "raw_response_ref": raw_ref, "raw_response_sha256": raw_sha,
            "parsed_result": parsed, "normalized_prediction": normalized,
            "schema_valid": normalized is not None, "format_repairs": repairs,
            "error_class": error_class.value,
            "error": self._public_error(error), "evaluation": evaluation,
            "input_tokens": usage["input"], "output_tokens": usage["output"],
            "reasoning_tokens": usage["reasoning"],
            "latency_ms": usage["latency"], "provider_attempts": usage["attempts"],
            "finish_reason": responses[-1].finish_reason if responses else None,
            "content_length": len(responses[-1].raw_text) if responses else 0,
            "cost_cny_delta": max(0.0, float(after["spent_cny"]) - float(record_ledger_before["spent_cny"])) if self.mode == "real" else 0.0,
            "cache_hit": bool(responses) and all(item.cached for item in responses),
            "fallback_used": False, "main_state_pollution": False,
            "leakage": {"passed": error_class != BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE, "api_key_value_present": False, "hidden_fields": []},
            "termination_reason": "COMPONENT_COMPLETE" if error is None else error_class.value,
        }
        self._assert_no_secrets(record)
        return record

    def _provider(self, model: ModelConfig) -> OpenAICompatibleProvider:
        key = model_api_key(model)
        if not key:
            raise PermissionError(f"{model.provider} credential is unconfigured")
        self.provider_instances += 1
        return OpenAICompatibleProvider(
            model, key, rate_limiter=RequestRateLimiter(1.0), cost_controller=self.costs,
        )

    def _cache(self, provider: str) -> LLMCache:
        directory = self.root / "cache" / provider / self.config.raw["execution"]["cache_namespace"]
        identity_path = directory / "identity.json"
        identity = {
            "schema_version": "specdriftbench-cache-identity-v1",
            "attempt_id": self.attempt_id, "provider": provider,
            "config_hash": self.config.config_hash, "source_snapshot": _source_snapshot(),
            "provider_capability_profile": ProviderCapabilityAdapter.for_model(
                self.config.model(provider)
            ).public_profile(),
        }
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Cache identity mismatch")
        if not identity_path.exists() and self.mode == "replay":
            raise FileNotFoundError(f"missing real Cache identity for {provider}")
        if not identity_path.exists():
            _atomic_json(identity_path, identity)
        return LLMCache(directory)

    def _write_or_validate_manifest(self) -> None:
        manifest = self._manifest()
        path = self.results / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("manifest identity changed")
        self.manifest_validator.validate(manifest)
        _atomic_json(path, manifest)

    def _manifest(self) -> dict[str, Any]:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        return {
            "schema_version": "specdriftbench-component-manifest-v2",
            "attempt_id": self.attempt_id, "git_commit": commit,
            "source_snapshot": _source_snapshot(), "config_hash": self.config.config_hash,
            "prompt_sha256": _sha256_bytes(PROMPT_PATH),
            "output_schema_sha256": _sha256_bytes(OUTPUT_SCHEMA_PATH),
            "evidence_view_definition_sha256": _sha256_bytes(VIEW_DEFINITION_PATH),
            "tool_registry_sha256": self.registry.fingerprint(),
            "tool_registry": self.registry.to_dict(),
            "provider_capability_profiles": {
                model.provider: ProviderCapabilityAdapter.for_model(model).public_profile()
                for model in self.config.models
            },
            "plan": [item.to_dict() for item in self.config.plan],
            "heldout48_overlap": 0,
            "ledger_before": self.ledger_before,
            "ledger_before_sha256": self.ledger_before_hash,
            "provider_order": ["deepseek", "dashscope", "moonshot"],
            "cache_namespaces": {provider: f"{provider}/{self.config.raw['execution']['cache_namespace']}" for provider in ("deepseek", "dashscope", "moonshot")},
            "result_namespace": self.config.raw["execution"]["result_namespace"],
        }

    def _baseline_from_manifest_or_current(self) -> tuple[dict[str, Any], str]:
        attempt_baseline = self.root / "attempt_ledger_baseline.json"
        if attempt_baseline.exists():
            value = json.loads(attempt_baseline.read_text())
            baseline = _public_ledger_snapshot(value["ledger_before"])
            current, _ = capture_ledger_baseline(self.ledger_path)
            if not _ledger_dominates(current, baseline):
                raise ValueError("formal Ledger rolled back below preflight baseline")
            return baseline, value["ledger_before_sha256"]
        for name in ("results", "replay"):
            path = self.root / name / "manifest.json"
            if path.exists():
                value = json.loads(path.read_text())
                baseline = _public_ledger_snapshot(value["ledger_before"])
                current, _ = capture_ledger_baseline(self.ledger_path)
                if not _ledger_dominates(current, baseline):
                    raise ValueError("formal Ledger rolled back below attempt manifest")
                return baseline, value["ledger_before_sha256"]
        return capture_ledger_baseline(self.ledger_path)

    def _settled_attempt_cost(self) -> float:
        total = 0.0
        preflight = self.root / "preflight/result.json"
        if preflight.exists(): total += float(json.loads(preflight.read_text()).get("cost_cny_delta", 0.0))
        for path in (self.root / "results/checkpoint").glob("*/*.json"):
            total += float(json.loads(path.read_text()).get("cost_cny_delta", 0.0))
        return total

    def _check_credentials(self) -> None:
        missing = [model.provider for model in self.config.models if not model_api_key(model)]
        if missing:
            raise PermissionError("unconfigured Provider credentials: " + ", ".join(missing))

    def _assert_no_secrets(self, value: Any) -> None:
        for model in self.config.models:
            key = model_api_key(model)
            if key: assert_secret_absent(value, key)

    @staticmethod
    def _evaluate(prediction: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> dict[str, bool | None]:
        if prediction is None:
            return {"class_correct": None, "category_correct": None, "target_correct": None, "location_correct": None}
        location = prediction.get("normalized_location") or {}
        expected_location = expected.get("location")
        location_correct = True if expected_location is None else expected_location in {location.get("spec_pointer"), location.get("runtime_path")}
        return {
            "class_correct": prediction.get("predicted_class") == expected["predicted_class"],
            "category_correct": prediction.get("drift_category") == expected["drift_category"],
            "target_correct": prediction.get("target_tool") == expected["target_tool"],
            "location_correct": location_correct,
        }

    @staticmethod
    def _public_error(error: BaseException | None) -> dict[str, Any] | None:
        if error is None: return None
        if isinstance(error, ProviderError): return error.public_dict()
        return {"type": type(error).__name__, "sanitized_message": str(error)[:512]}

    @staticmethod
    def _is_infrastructure(record: Mapping[str, Any]) -> bool:
        return record["error_class"] in {
            BenchmarkErrorClass.EVALUATOR_ERROR.value,
            BenchmarkErrorClass.CONTROLLER_ERROR.value,
            BenchmarkErrorClass.NETWORK_ERROR.value,
            BenchmarkErrorClass.PROVIDER_ERROR.value,
        }

    def _write_summary(self, records: list[dict[str, Any]], status: str) -> dict[str, Any]:
        current, digest = capture_ledger_baseline(self.ledger_path)
        metrics = aggregate_metrics(records, self.builder)
        delta_baseline = self.ledger_before if self.mode == "real" else self.invocation_before
        summary = {
            "schema_version": "specdriftbench-component-summary-v1",
            "attempt_id": self.attempt_id, "mode": self.mode, "status": status,
            "records_planned": 36, "records_completed": len(records),
            "provider_instances": self.provider_instances if self.mode == "real" else 0,
            "network_calls": self.network_calls if self.mode == "real" else 0,
            "ledger_before": self.ledger_before, "ledger_before_sha256": self.ledger_before_hash,
            "ledger_after": current, "ledger_after_sha256": digest,
            "attempts_delta": int(current["api_attempts"] - delta_baseline["api_attempts"]),
            "input_tokens_delta": int(current["provider_reported_input_tokens"] - delta_baseline["provider_reported_input_tokens"]),
            "output_tokens_delta": int(current["provider_reported_output_tokens"] - delta_baseline["provider_reported_output_tokens"]),
            "cost_cny_delta": float(current["spent_cny"] - delta_baseline["spent_cny"]),
            "metrics": metrics,
            "fallback_count": sum(item["fallback_used"] for item in records),
            "main_state_pollution_count": sum(item["main_state_pollution"] for item in records),
            "true_leakage_count": sum(item["error_class"] == BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE.value for item in records),
        }
        _atomic_json(self.results / "summary.json", summary)
        return summary


class CacheReplayMiss(RuntimeError):
    pass


def aggregate_metrics(records: list[dict[str, Any]], builder: EvidenceViewBuilder) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    view_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        expected = expected_for_evaluator(builder, record["family"], record["variant"])["predicted_class"]
        prediction = record.get("normalized_prediction") or {}
        predicted = prediction.get("predicted_class") or "INVALID"
        confusion[expected][predicted] += 1
        view_rows[record["evidence_view"]].append((expected, predicted))
    return {
        "schema_valid_rate": _rate(sum(item["schema_valid"] for item in records), len(records)),
        "infrastructure_error_rate": _rate(sum(SpecDriftBenchRunner._is_infrastructure(item) for item in records), len(records)),
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "macro_f1_by_evidence_view": {key: _macro_f1(rows) for key, rows in view_rows.items()},
        "class_accuracy": _rate(sum(item["evaluation"]["class_correct"] is True for item in records), len(records)),
        "category_accuracy": _rate(sum(item["evaluation"]["category_correct"] is True for item in records), len(records)),
        "target_accuracy_pd": _conditional_rate(records, "target_correct"),
        "location_accuracy_pd": _conditional_rate(records, "location_correct"),
        "abstention_rate": _rate(sum(bool((item.get("normalized_prediction") or {}).get("abstain")) for item in records), len(records)),
        "invalid_output_count": sum(not item["schema_valid"] for item in records),
        "format_repair_count": sum(item["format_repairs"] for item in records),
        "input_tokens": sum(item["input_tokens"] for item in records),
        "output_tokens": sum(item["output_tokens"] for item in records),
        "latency_ms": sum(item["latency_ms"] for item in records),
        "provider_attempts": sum(item["provider_attempts"] for item in records),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _conditional_rate(records: list[dict[str, Any]], key: str) -> float | None:
    rows = [item for item in records if item["variant"] == "PD" and item["schema_valid"]]
    return _rate(sum(item["evaluation"][key] is True for item in rows), len(rows))


def _macro_f1(rows: list[tuple[str, str]]) -> float | None:
    if not rows: return None
    labels = ("AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT")
    scores = []
    for label in labels:
        tp = sum(expected == label and predicted == label for expected, predicted in rows)
        fp = sum(expected != label and predicted == label for expected, predicted in rows)
        fn = sum(expected == label and predicted != label for expected, predicted in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def run_kimi_preflight(
    config_path: Path | str, *, attempt_id: str, allow_real_api: bool,
    confirmation: str | None, ledger_path: Path = LEDGER,
    output_root: Path | None = None,
) -> dict[str, Any]:
    config = SpecDriftBenchConfig.load(config_path)
    if not (config.authorized and allow_real_api and confirmation == CONFIRM_PREFLIGHT):
        raise PermissionError("Kimi preflight requires authorized config, --allow-real-api and RUN-KIMI-PREFLIGHT-1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id):
        raise ValueError("unsafe attempt ID")
    model = replace(config.model("moonshot"), max_provider_retries=0)
    base = PROJECT_ROOT / "results/experiments/phase10/specdriftbench_component_canary/attempts" / attempt_id
    root = Path(output_root or base)
    result_path = root / "preflight/result.json"
    if result_path.exists():
        raise FileExistsError("Kimi preflight result already exists; refusing a second real call")
    key = model_api_key(model)
    if not key:
        baseline, baseline_hash = capture_ledger_baseline(Path(ledger_path))
        result = {
            "schema_version": "specdriftbench-kimi-preflight-v1",
            "attempt_id": attempt_id, "provider": "moonshot",
            "configured_model": model.model_id, "endpoint": model.base_url,
            "credential": "unconfigured", "response_format": "json_object",
            "passed": False, "parsed_result": None, "http_status": None,
            "error_class": BenchmarkErrorClass.PROVIDER_ERROR.value,
            "error": {"type": "CredentialMissing", "sanitized_message": "MOONSHOT_API_KEY is unconfigured"},
            "request_id": None, "retryable": False, "network_attempts": 0,
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0,
            "cost_cny_delta": 0.0, "ledger_before": baseline,
            "ledger_after": baseline, "ledger_before_sha256": baseline_hash,
            "cache_key": None, "cache_entry": False, "api_key_leakage": False,
        }
        _atomic_json(result_path, result)
        return result
    baseline, baseline_hash = capture_ledger_baseline(Path(ledger_path))
    _atomic_json(root / "attempt_ledger_baseline.json", {
        "attempt_id": attempt_id, "ledger_before": baseline,
        "ledger_before_sha256": baseline_hash,
    })
    pricing = PricingCatalog(json.loads(PRICING_PATH.read_text(encoding="utf-8")))
    costs = AtomicFocusedLedger(
        Path(ledger_path), _LedgerConfigAdapter(config), attempt_id, baseline,
        pricing_catalog=pricing,
    )
    cache_dir = root / "preflight/cache/specdriftbench_kimi_preflight_v1"
    _atomic_json(cache_dir / "identity.json", {
        "schema_version": "specdriftbench-kimi-preflight-cache-v1",
        "attempt_id": attempt_id, "provider": "moonshot", "model": "kimi-k2.6",
        "config_hash": config.config_hash,
        "provider_capability_profile": ProviderCapabilityAdapter.for_model(model).public_profile(),
    })
    cache = LLMCache(cache_dir)
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["ok", "provider", "model"],
        "properties": {
            "ok": {"const": True}, "provider": {"const": "moonshot"},
            "model": {"const": "kimi-k2.6"},
        },
    }
    messages = (
        {"role": "system", "content": "Return only one valid JSON object."},
        {"role": "user", "content": 'Return exactly {"ok":true,"provider":"moonshot","model":"kimi-k2.6"}.'},
    )
    request = ProviderRequest(
        messages, schema, hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "public-kimi-preflight-v1", 1, "diagnostic_preflight", 0,
        config.raw["execution"]["seed"], "specdriftbench_preflight", config.config_hash,
    )
    cache_key = LLMCache.key(model, request, hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest())
    before = json.loads(Path(ledger_path).read_text())
    response = None
    error: BaseException | None = None
    parsed = None
    try:
        provider = OpenAICompatibleProvider(
            model, key, rate_limiter=RequestRateLimiter(1.0), cost_controller=costs,
        )
        response = provider.complete(request)
        cache.put(cache_key, response, key)
        assert_benchmark_visible(response.raw_text)
        if response.finish_reason == "length":
            raise OutputTruncated(response.finish_reason, len(response.raw_text))
        parsed = parse_structured_output(response.raw_text, schema)
        assert_benchmark_visible(parsed)
        if response.input_tokens <= 0 or response.total_tokens <= 0:
            raise ValueError("Kimi usage was not returned")
        if response.model != model.model_id:
            raise ValueError("Kimi actual model does not match configured model")
    except BaseException as exc:
        error = exc
    after = json.loads(Path(ledger_path).read_text())
    public_error = SpecDriftBenchRunner._public_error(error)
    result = {
        "schema_version": "specdriftbench-kimi-preflight-v1",
        "attempt_id": attempt_id, "provider": "moonshot", "configured_model": model.model_id,
        "endpoint": model.base_url, "credential": "configured",
        "response_format": "json_object", "passed": error is None,
        "parsed_result": parsed,
        "http_status": 200 if response is not None else getattr(error, "status_code", None),
        "error_class": BenchmarkErrorClass.NONE.value if error is None else classify_error(error).value,
        "error": public_error,
        "request_id": response.response_id if response else getattr(error, "request_id", None),
        "retryable": bool(getattr(error, "retryable", False)),
        "network_attempts": response.provider_attempts if response else int(getattr(error, "actual_network_attempts", 0)),
        "input_tokens": response.input_tokens if response else 0,
        "output_tokens": response.output_tokens if response else 0,
        "reasoning_tokens": response.reasoning_tokens if response else 0,
        "usage": {
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "total_tokens": response.total_tokens if response else 0,
            "reasoning_tokens": response.reasoning_tokens if response else 0,
        },
        "finish_reason": response.finish_reason if response else None,
        "content_length": len(response.raw_text) if response else 0,
        "max_tokens": model.max_output_tokens,
        "latency_ms": response.latency_ms if response else 0.0,
        "cost_cny_delta": max(0.0, float(after["spent_cny"]) - float(before["spent_cny"])),
        "ledger_before": _public_ledger_snapshot(before),
        "ledger_after": _public_ledger_snapshot(after),
        "cache_key": cache_key, "cache_entry": cache.get(cache_key) is not None,
        "provider_capability_profile": ProviderCapabilityAdapter.for_model(model).public_profile(),
        "provider_profile_id": ProviderCapabilityAdapter.for_model(model).profile_id,
        "api_key_leakage": False,
    }
    assert_secret_absent(result, key)
    _atomic_json(result_path, result)
    return result


def prepare_offline_assets(
    output: Path | None = None, config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    config = SpecDriftBenchConfig.load(config_path)
    builder = EvidenceViewBuilder()
    registry = CanonicalToolRegistry.from_displayed_spec()
    base = Path(output or PROJECT_ROOT / "results/experiments/phase10/specdriftbench_component_canary/offline_preparation_v1")
    hashes: dict[str, list[str]] = defaultdict(list)
    for row in builder.plan():
        evidence = builder.build(row["family"], row["variant"], row["evidence_view"])
        name = f"{row['family']}-{row['variant']}-{row['evidence_view']}.json"
        _atomic_json(base / "evidence" / name, evidence)
        hashes[row["evidence_view"]].append(evidence["evidence_sha256"])
    view_hashes = {
        view: hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()
        for view, values in hashes.items()
    }
    _atomic_json(base / "tool_registry_v1.json", registry.to_dict())
    baseline, baseline_hash = capture_ledger_baseline()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout
    report = {
        "schema_version": "specdriftbench-offline-preparation-v1",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "workspace_clean": not bool(status.strip()),
        "workspace_expected_experiment_assets_only": all(
            line[3:].startswith(("results/experiments/phase10/cost/ledger.json", "results/experiments/phase10/driftguard_focused_canary/real_attempts/"))
            for line in status.splitlines() if line.strip()
        ),
        "ledger": baseline, "ledger_sha256": baseline_hash,
        "records_planned": 36, "heldout48_overlap": 0,
        "providers": [model.public_dict() for model in config.models],
        "credential_status": {model.provider: "configured" if model_api_key(model) else "unconfigured" for model in config.models},
        "balanced_plan": [item.to_dict() for item in config.plan],
        "counts": {
            "providers": Counter(item.provider for item in config.plan),
            "variants_per_provider": {provider: Counter(item.variant for item in config.plan if item.provider == provider) for provider in ("deepseek", "dashscope", "moonshot")},
            "views_per_provider": {provider: Counter(item.evidence_view for item in config.plan if item.provider == provider) for provider in ("deepseek", "dashscope", "moonshot")},
        },
        "view_definition_sha256": _sha256_bytes(VIEW_DEFINITION_PATH),
        "view_content_sha256": view_hashes,
        "tool_registry_sha256": registry.fingerprint(),
        "prompt_sha256": _sha256_bytes(PROMPT_PATH),
        "schema_sha256": _sha256_bytes(OUTPUT_SCHEMA_PATH),
    }
    _atomic_json(base / "report.json", report)
    return report


def offline_rescore_v1(
    source_attempt: Path = OLD_ATTEMPT, output: Path | None = None,
) -> dict[str, Any]:
    records_root = source_attempt / "results/records"
    cache_root = source_attempt / "cache"
    records = [json.loads(path.read_text()) for path in records_root.glob("*.json")]
    registry = CanonicalToolRegistry.from_displayed_spec()
    raw_by_sha: dict[str, str] = {}
    for path in cache_root.rglob("raw/*.json"):
        value = json.loads(path.read_text())
        raw = value.get("raw_text")
        if isinstance(raw, str): raw_by_sha[hashlib.sha256(raw.encode()).hexdigest()] = raw
    rows = []
    counts = Counter()
    for record in sorted(records, key=lambda item: ({"deepseek": 0, "dashscope": 1}[item["provider"]], FAMILIES.index(item["family"]))):
        candidate = record.get("final_attribution") or record.get("preliminary_attribution")
        recovered = None
        legal_label_false_positive = False
        for call in reversed(record.get("llm_stage_calls", [])):
            raw = raw_by_sha.get(call.get("raw_response_sha256"))
            if raw:
                try:
                    recovered = json.loads(raw)
                    assert_benchmark_visible(recovered)
                except TrueEvidenceLeakage:
                    recovered = None
                except (ValueError, TypeError):
                    recovered = None
                if record["termination_reason"] == "EvidenceLeakageError" and recovered is not None:
                    legal_label_false_positive = True
                break
        source = candidate or recovered
        converted = None
        if isinstance(source, Mapping):
            converted = {
                "predicted_class": source.get("predicted_class"),
                "drift_category": "NONE" if source.get("drift_category") in {None, "UNKNOWN"} else source.get("drift_category"),
                "target_tool": source.get("target_tool_id"),
                "normalized_location": source.get("location"),
                "confidence": source.get("confidence"),
                "abstain": source.get("predicted_class") == "INSUFFICIENT_EVIDENCE",
                "concise_reason": source.get("concise_reason"),
            }
            converted = normalize_prediction(converted, registry)
        if legal_label_false_positive:
            revised_error = BenchmarkErrorClass.EVALUATOR_ERROR.value
            counts["evaluator_errors"] += 1
        elif record["termination_reason"] == "BudgetExhausted":
            revised_error = BenchmarkErrorClass.METHOD_BUDGET_FAILURE.value
            counts["budget_errors"] += 1
        elif record["termination_reason"] == "CONTROLLER_PROBE_ERROR":
            revised_error = BenchmarkErrorClass.CONTROLLER_ERROR.value
            counts["controller_errors"] += 1
        else:
            revised_error = BenchmarkErrorClass.NONE.value if candidate is not None else BenchmarkErrorClass.MODEL_ERROR.value
        invalid = any(call.get("validation_error") for call in record.get("llm_stage_calls", []))
        if invalid:
            counts["model_output_schema_errors"] += 1
            counts["model_errors"] += 1
        rows.append({
            "provider": record["provider"], "family": record["family"],
            "old_termination": record["termination_reason"], "reclassified_error": revised_error,
            "legal_prediction_label_false_positive": legal_label_false_positive,
            "original_parsed_attribution_available": candidate is not None,
            "raw_json_recovered": recovered is not None,
            "usable_attribution_before_budget_exhaustion": record["termination_reason"] == "BudgetExhausted" and candidate is not None,
            "original_target_tool": (source or {}).get("target_tool_id") if isinstance(source, Mapping) else None,
            "canonical_target_tool": converted.get("target_tool") if converted else None,
            "tool_id_normalized": bool(source and (source.get("target_tool_id") != converted.get("target_tool"))) if converted else False,
            "rescored_prediction": converted,
        })
    report = {
        "schema_version": "specdriftbench-offline-rescore-v1",
        "source_attempt": source_attempt.name, "source_records": len(records),
        "real_api_calls": 0, "old_records_modified": False,
        "tool_registry_sha256": registry.fingerprint(),
        "counts": dict(counts), "records": rows,
    }
    target = Path(output or source_attempt / "offline_rescore_v1.json")
    _atomic_json(target, report)
    lines = [
        "# SpecDriftBench offline rescore v1", "",
        f"- Source records: {len(records)}; real API calls: 0; old records modified: false.",
        f"- Legal-label evaluator false positives: {counts['evaluator_errors']}.",
        f"- Method budget failures: {counts['budget_errors']}.",
        f"- Model output schema errors observed before repair/failure: {counts['model_output_schema_errors']}.",
        "- T03 and canonical operationId values are evaluated through the frozen bidirectional Tool Registry.",
    ]
    _atomic_text(target.with_suffix(".md"), "\n".join(lines) + "\n")
    return report
