from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping

from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.budgets import BudgetExhausted
from driftguard.experiments.checkpoint import CheckpointStore
from driftguard.live.focused_config import FocusedLiveHealingConfig, FocusedRecordPlan
from driftguard.live.provider_adapter import FocusedProviderAdapter, ReplayCacheMiss
from driftguard.llm import (
    InvalidStructuredOutput, LLMCache, LLMProvider, ModelConfig,
    ProviderError, ProviderRequest, ProviderResponse, sanitize_provider_text,
)
from driftguard.phase10.consistency_audit import source_snapshot
from driftguard.phase10.costs import CostBudgetManager, CostHardLimit, Reservation
from driftguard.phase10.credentials import credential_status
from driftguard.phase10.pricing import PricingCatalog

from .live_healing_mock_runner import LiveHealingMockRunner


LEDGER = PROJECT_ROOT / "results/experiments/phase10/cost/ledger.json"
RECORD_SCHEMA = PROJECT_ROOT / "benchmark/schemas/focused_live_healing_record_schema_v1.json"
MANIFEST_SCHEMA = PROJECT_ROOT / "benchmark/schemas/focused_live_healing_manifest_schema_v1.json"
REAL_CACHE_NAMESPACE = "driftguard_focused_live_healing_real_v1"
CONFIRMATION = "RUN-EXACTLY-8"
LEDGER_BASELINE = {
    "api_attempts": 863,
    "provider_reported_input_tokens": 3_580_832,
    "provider_reported_output_tokens": 84_182,
    "spent_cny": 5.202644120,
}
ProviderFactory = Callable[
    [ModelConfig, Any, bool, Any], LLMProvider
]


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _focused_source_snapshot() -> str:
    paths = (
        "src/driftguard/agents/controller.py",
        "src/driftguard/live/orchestrator.py",
        "src/driftguard/live/stages.py",
        "src/driftguard/live/provider_adapter.py",
        "src/driftguard/runners/live_healing_mock_runner.py",
        "src/driftguard/runners/focused_live_healing_runner.py",
    )
    values = {path: _sha256_bytes(PROJECT_ROOT / path) for path in paths}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class FocusedCache:
    """A hash-bound namespace that cannot alias an earlier or Mock cache."""

    def __init__(
        self, root: Path, namespace: str, identity: dict[str, Any],
        *, read_only: bool = False,
    ) -> None:
        if namespace in {"v1", "v2", "v3", "v4", "v4b"}:
            raise ValueError("Focused Canary cannot use a legacy cache namespace")
        self.namespace = namespace
        self.directory = root / namespace
        self.identity_path = self.directory / "focused_cache_identity.json"
        if self.identity_path.exists():
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise ValueError("Focused cache identity changed; refusing cache reuse")
        elif read_only:
            raise FileNotFoundError("real Cache identity is missing; replay-real is read-only")
        else:
            _atomic_json(self.identity_path, identity)
        if read_only and not all((self.directory / name).is_dir() for name in ("raw", "parsed")):
            raise FileNotFoundError("real Cache raw/parsed directories are missing")
        self.cache = LLMCache(self.directory)

    def raw_count(self) -> int:
        return len(tuple((self.directory / "raw").glob("*.json")))

    def assert_no_mock_entries(self) -> None:
        kind = json.loads(self.identity_path.read_text(encoding="utf-8")).get("cache_kind")
        if kind not in {"REAL_PROVIDER", "FAKE_PROVIDER_TEST"}:
            raise ValueError("real Cache identity is missing a real-path cache marker")
        for path in (self.directory / "raw").glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            model = str(value.get("model", "")).lower()
            response_id = str(value.get("response_id", "")).lower()
            if model.startswith("mock-") or response_id.startswith("mock-"):
                raise ValueError("real Cache contains a Mock Provider entry")


class FocusedResultWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.records = directory / "records"
        self.record_validator = Draft202012Validator(
            json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
        )
        self.manifest_validator = Draft202012Validator(
            json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
        )

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_validator.validate(manifest)
        path = self.directory / "manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Focused result manifest changed; refusing overwrite")
        _atomic_json(path, manifest)

    def write_record(self, record: dict[str, Any], *, allow_existing: bool = False) -> None:
        self.record_validator.validate(record)
        path = self.records / f"{record['record_id']}.json"
        if path.exists() and not allow_existing:
            raise FileExistsError(f"Focused result already exists: {record['record_id']}")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Focused result differs from checkpoint: {record['record_id']}")
        _atomic_json(path, record)

    def write_summary(self, summary: dict[str, Any]) -> None:
        _atomic_json(self.directory / "summary.json", summary)

    def write_deepseek_gate(self, report: dict[str, Any]) -> None:
        _atomic_json(self.directory / "deepseek_gate_report.json", report)


class AtomicFocusedLedger:
    """Existing reservation manager with an atomic persistent Focused ledger view."""

    def __init__(
        self, path: Path, config: FocusedLiveHealingConfig, attempt_id: str,
        attempt_baseline: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.attempt_id = attempt_id
        self.config_hash = config.config_hash
        self.original = json.loads(path.read_text(encoding="utf-8"))
        self.initial = {
            "spent_cny": float(self.original["spent_cny"]),
            "api_attempts": int(self.original["api_attempts"]),
            "provider_reported_input_tokens": int(self.original["provider_reported_input_tokens"]),
            "provider_reported_output_tokens": int(self.original["provider_reported_output_tokens"]),
        }
        self.attempt_initial = dict(attempt_baseline or self.initial)
        execution, budget = config.raw["execution"], config.raw["budget"]
        soft = float(self.attempt_initial["spent_cny"]) + float(execution["attempt_soft_increment_cny"])
        hard = min(
            float(self.attempt_initial["spent_cny"]) + float(execution["attempt_hard_increment_cny"]),
            float(execution["total_hard_limit_cny"]),
            float(budget["full_hard_limit_cny"]),
        )
        self.manager = CostBudgetManager(
            PricingCatalog.load_default(), soft, hard,
            int(budget["max_input_tokens_per_record"]),
            int(budget["max_output_tokens_per_record"]),
            self.initial["spent_cny"], self.initial["api_attempts"],
            self.initial["provider_reported_input_tokens"],
            self.initial["provider_reported_output_tokens"],
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.manager.events

    def reserve(self, config: ModelConfig, request: ProviderRequest) -> Reservation:
        reservation = self.manager.reserve(config, request)
        self._persist()
        return reservation

    def settle(self, reservation: Reservation, response: ProviderResponse) -> None:
        self.manager.settle(reservation, response)
        self._persist()

    def release(self, reservation: Reservation) -> None:
        self.manager.release(reservation)
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        return self.manager.snapshot()

    def _persist(self) -> None:
        snapshot = self.snapshot()
        value = {
            **self.original,
            **snapshot,
            "attempt": self.attempt_id,
            "config_hash": self.config_hash,
            "attempt_base_spent_cny": self.attempt_initial["spent_cny"],
            "attempt_incremental_spent_cny": snapshot["spent_cny"] - float(self.attempt_initial["spent_cny"]),
            "incremental_soft_limit_cny": snapshot["soft_limit_cny"] - float(self.attempt_initial["spent_cny"]),
            "incremental_hard_limit_cny": snapshot["hard_limit_cny"] - float(self.attempt_initial["spent_cny"]),
            "total_hard_limit_cny": 50.0,
        }
        _atomic_json(self.path, value)
        self.original = value


class FocusedLiveHealingRunner:
    """Dedicated exact-eight runner; never enters generic Phase 10 enumeration."""

    MODES = {"mock", "replay", "real", "replay-real"}

    def __init__(
        self,
        config_path: Path | str,
        *,
        mode: str,
        output: Path | None = None,
        cache_root: Path | None = None,
        attempt_id: str | None = None,
        allow_real_api: bool = False,
        confirm_focused_canary: str | None = None,
        provider_factory: ProviderFactory | None = None,
        credential_checker: Callable[[], Mapping[str, str]] = credential_status,
        ledger_path: Path = LEDGER,
        git_state_provider: Callable[[], dict[str, Any]] = _git_state,
        secret_values: tuple[str, ...] = (),
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported Focused runner mode: {mode}")
        self.config = FocusedLiveHealingConfig.load(config_path)
        if len(self.config.plan) != 8:
            raise ValueError("Focused runner refuses any plan other than exactly 8 records")
        self.mode = mode
        self.attempt_id = attempt_id
        self.allow_real_api = allow_real_api
        self.confirmation = confirm_focused_canary
        self.provider_factory = provider_factory
        self.provider_kind = "FAKE_PROVIDER_TEST" if provider_factory else "REAL_PROVIDER"
        self.ledger_path = Path(ledger_path)
        self.git_state = git_state_provider()
        self._secret_values = tuple(value for value in secret_values if value)
        self.snapshot = source_snapshot()
        self.source_hash = self.snapshot["source_snapshot_hash"]
        self.resumed_records = 0
        self.real_provider_instances = 0
        self.network_calls = 0
        self.mock_runner = LiveHealingMockRunner()

        if mode in {"mock", "replay"}:
            if self.config.run_authorized:
                raise PermissionError("offline Mock modes require run_authorized:false")
            base = output or self.config.output_directory / mode
            root = cache_root or self.config.output_directory / "cache"
            namespace = self.config.raw["execution"]["cache_namespace"]
        else:
            self._validate_attempt_id()
            attempt_root = self.config.output_directory / "real_attempts" / str(attempt_id)
            base = output or attempt_root / ("results" if mode == "real" else "replay")
            root = cache_root or attempt_root / "cache"
            namespace = REAL_CACHE_NAMESPACE
            self.source_hash = _focused_source_snapshot()
            if mode == "real":
                self._validate_real_authorization(credential_checker)
            elif not self.config.run_authorized:
                raise PermissionError("replay-real requires the authorized config used by the real attempt")

        self.output = Path(base)
        self.cache_root = Path(root)
        if mode == "real":
            self._validate_real_paths(namespace)
        if mode == "replay-real":
            identity_path = self.cache_root / namespace / "focused_cache_identity.json"
            if not identity_path.exists():
                raise FileNotFoundError("specified real attempt Cache does not exist")
            existing_kind = json.loads(identity_path.read_text(encoding="utf-8")).get("cache_kind")
            if existing_kind not in {"REAL_PROVIDER", "FAKE_PROVIDER_TEST"}:
                raise ValueError("replay-real refuses a Mock or unmarked Cache")
            self.provider_kind = existing_kind

        self.freeze_identity = self._freeze_identity(namespace)
        self.cache_identity_hash = hashlib.sha256(json.dumps(
            self.freeze_identity, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        self._validate_ledger_preflight()
        self.cache_store = FocusedCache(
            self.cache_root, namespace, self.freeze_identity,
            read_only=mode == "replay-real",
        )
        if mode in {"real", "replay-real"}:
            self.cache_store.assert_no_mock_entries()
        self.writer = FocusedResultWriter(self.output)
        self.checkpoints = CheckpointStore(self.output / "checkpoint")
        self.checkpoint_identity_path = self.output / "checkpoint/identity.json"
        self.costs = (
            AtomicFocusedLedger(
                self.ledger_path, self.config, str(attempt_id),
                attempt_baseline=self._existing_attempt_baseline(),
            )
            if mode == "real" else None
        )
        self.run_ledger_before = self._read_ledger()
        self.run_ledger_before_hash = _sha256_bytes(self.ledger_path)

    @staticmethod
    def validate_config(config_path: Path | str) -> dict[str, Any]:
        config = FocusedLiveHealingConfig.load(config_path)
        return {
            "experiment_kind": config.experiment_kind,
            "records_planned": len(config.plan),
            "heldout48_overlap": 0,
            "run_authorized": config.run_authorized,
            "plan": config.plan_dicts(),
        }

    @classmethod
    def dry_run(cls, config_path: Path | str) -> dict[str, Any]:
        report = cls.validate_config(config_path)
        return {**report, "execution": "DRY_RUN", "network_calls": 0, "provider_instances": 0}

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        self._bind_checkpoint(resume)
        manifest = self._manifest()
        self.writer.write_manifest(manifest)
        completed: list[dict[str, Any]] = []
        self.writer.write_summary(self._summary(completed, "INCOMPLETE"))
        try:
            for plan in self.config.plan:
                if self.checkpoints.has(plan.identity):
                    if not resume:
                        raise FileExistsError(f"checkpoint exists for {plan.identity}; pass --resume")
                    record = self.checkpoints.read(plan.identity)
                    self._verify_resumed_record(plan, record)
                    self.writer.write_record(record, allow_existing=True)
                    self.resumed_records += 1
                else:
                    record = self._run_record(plan)
                    self.writer.write_record(record)
                    self.checkpoints.write(plan.identity, record)
                completed.append(record)
                self.writer.write_summary(self._summary(completed, "INCOMPLETE"))
                if plan.ordinal == 4 and self.mode in {"real", "replay-real"}:
                    gate = self._deepseek_gate(completed)
                    self.writer.write_deepseek_gate(gate)
                    if not gate["passed"]:
                        summary = self._summary(completed, "BLOCKED_AFTER_DEEPSEEK")
                        summary["deepseek_gate"] = gate
                        self.writer.write_summary(summary)
                        return {"manifest": manifest, "summary": summary, "records": completed}
        except Exception as exc:
            failed = self._summary(completed, "INCOMPLETE")
            failed["stopped_reason"] = f"{type(exc).__name__}: {exc}"
            self.writer.write_summary(failed)
            raise
        if len(completed) != 8:
            raise RuntimeError(f"Focused execution must complete exactly 8 records, got {len(completed)}")
        if self.mode != "real" and _sha256_bytes(self.ledger_path) != self.run_ledger_before_hash:
            raise RuntimeError("formal ledger changed during a non-real Focused execution")
        final = self._summary(completed, "COMPLETE")
        final["ledger_after_sha256"] = _sha256_bytes(self.ledger_path)
        self.writer.write_summary(final)
        return {"manifest": manifest, "summary": final, "records": completed}

    def _run_record(self, plan: FocusedRecordPlan) -> dict[str, Any]:
        model = next(
            item for item in self.config.models
            if (item.provider, item.model_id) == (plan.provider, plan.model_id)
        )
        model = replace(model, seed=plan.seed)
        adapter_mode = "replay" if self.mode in {"replay", "replay-real"} else self.mode
        adapter = FocusedProviderAdapter(
            adapter_mode, model,
            allow_real_api=self.allow_real_api,
            confirm_focused_canary=self.confirmation == CONFIRMATION,
            run_authorized=self.config.run_authorized,
            cost_controller=self.costs,
            provider_factory=self.provider_factory,
        )
        cache_before = self.cache_store.raw_count()
        ledger_before = self._read_ledger()
        event_before = len(self.costs.events) if self.costs else 0
        try:
            raw = self.mock_runner.run_pd(
                plan.family, model_config=model, provider_adapter=adapter,
                cache=self.cache_store.cache, config_hash=self.cache_identity_hash,
                repetition=plan.repetition, seed=plan.seed,
            )
        except ReplayCacheMiss:
            raise
        except InvalidStructuredOutput as exc:
            return self._failure_record(
                plan, "METHOD_FAILURE", "INVALID_STRUCTURED_OUTPUT", exc,
                cache_before, adapter.provider_calls, ledger_before, event_before,
            )
        except (ProviderError, CostHardLimit, BudgetExhausted) as exc:
            return self._failure_record(
                plan, "INFRASTRUCTURE_ERROR", type(exc).__name__, exc,
                cache_before, adapter.provider_calls, ledger_before, event_before,
            )
        except Exception as exc:
            return self._failure_record(
                plan, "INFRASTRUCTURE_ERROR", type(exc).__name__, exc,
                cache_before, adapter.provider_calls, ledger_before, event_before,
            )
        self.real_provider_instances += len(adapter.created) if self.mode == "real" else 0
        self.network_calls += adapter.provider_calls if self.mode == "real" and self.provider_factory is None else 0
        ledger_after = self._read_ledger()
        events = self.costs.events[event_before:] if self.costs else []
        return self._record(
            plan, raw, cache_before, self.cache_store.raw_count(), adapter.provider_calls,
            ledger_before, ledger_after, events,
        )

    def _record(
        self, plan: FocusedRecordPlan, raw: dict[str, Any],
        cache_before: int, cache_after: int, provider_calls: int,
        ledger_before: dict[str, Any], ledger_after: dict[str, Any],
        provider_usage: list[dict[str, Any]],
    ) -> dict[str, Any]:
        termination = raw["termination_reason"]
        infrastructure_categories = {"PROVIDER_ERROR", "PROVIDER_TIMEOUT", "RATE_LIMITED"}
        controller_infrastructure_error = any(
            item.get("error_category") in infrastructure_categories
            for item in raw.get("controller_runs", [])
        )
        method_failure = termination != "LIVE_HEALING_COMPLETE"
        stage_calls = deepcopy(raw["llm_stage_calls"])
        if self.mode in {"mock", "replay", "replay-real"} or self.provider_kind == "FAKE_PROVIDER_TEST":
            for call in stage_calls:
                call["test_or_cached_provider_attempts"] = call.get("provider_attempts", 0)
                if self.mode != "real" or self.provider_kind == "FAKE_PROVIDER_TEST":
                    call["provider_attempts"] = 0
        input_delta = self._delta(ledger_before, ledger_after, "provider_reported_input_tokens")
        output_delta = self._delta(ledger_before, ledger_after, "provider_reported_output_tokens")
        attempt_delta = self._delta(ledger_before, ledger_after, "api_attempts")
        cost_delta = float(ledger_after["spent_cny"]) - float(ledger_before["spent_cny"])
        record = {
            "schema_version": "focused-live-healing-record-v1",
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": self._execution_mode(),
            "record_id": plan.identity,
            "record_status": "COMPLETE",
            "provider": plan.provider,
            "model": plan.model_id,
            "provider_kind": self.provider_kind if self.mode == "real" else "CACHE_ONLY" if "replay" in self.mode else "MOCK_PROVIDER",
            "attempt_id": self.attempt_id,
            "method": plan.method,
            "family": plan.family,
            "scenario_id": plan.scenario_id,
            "public_scenario_id": raw["public_scenario_id"],
            "source_snapshot": self.source_hash,
            "frozen_fingerprints": deepcopy(self.config.raw["fingerprints"]),
            "stage_trace": self._stage_trace(raw),
            "live_evidence_trace": raw["live_evidence_trace"],
            "evidence_provenance": {
                "source": "ToolAgentController live events",
                "trace_id": raw["live_evidence_trace"]["trace_id"],
                "event_count": len(raw["live_evidence_trace"]["events"]),
                "phase7_static_reconstruction": False,
            },
            "llm_stage_calls": stage_calls,
            "llm_calls": raw["llm_calls"],
            "tool_calls": raw["tool_calls"],
            "probe_calls": raw["probe_calls"],
            "patch_proposal_calls": raw["patch_proposal_calls"],
            "patch_revision": raw["patch_revision"],
            "preliminary_attribution": raw["preliminary_attribution"],
            "probe_selection": raw["probe_selection"],
            "probe_result": raw["probe_result"],
            "final_attribution": raw["final_attribution"],
            "patch_eligibility": raw["patch_eligibility"],
            "patch_proposal": raw["raw_llm_patch"],
            "deterministic_gates": {
                "static": raw["static_validation"],
                "regression": raw["regression_validation"],
                "safety": raw["safety_validation"],
                "minimality": raw["minimality_validation"],
            },
            "immediate_repair": raw["immediate_repair"],
            "future_transfer": raw["future_transfer"],
            "task_success": bool(raw["immediate_repair"] and raw["immediate_repair"].get("passed")),
            "termination_reason": termination,
            "stopped_stage": self._last_stage(raw),
            "infrastructure_or_method_failure": (
                "INFRASTRUCTURE_ERROR" if controller_infrastructure_error
                else "METHOD_FAILURE" if method_failure else "NONE"
            ),
            "input_tokens": input_delta,
            "output_tokens": output_delta,
            "latency_ms": sum(item.get("latency_ms", 0.0) for item in raw["llm_stage_calls"]),
            "api_attempts_delta": attempt_delta,
            "cost_cny_delta": cost_delta,
            "provider_boundary_calls": provider_calls,
            "provider_usage": deepcopy(provider_usage),
            "controller_runs": deepcopy(raw.get("controller_runs", [])),
            "provider_errors": self._controller_provider_errors(raw),
            "actual_network_attempts": attempt_delta,
            "ledger_before": self._public_ledger(ledger_before),
            "ledger_after": self._public_ledger(ledger_after),
            "cache": {
                "namespace": self.cache_store.namespace,
                "identity_hash": self.cache_identity_hash,
                "entries_before": cache_before,
                "entries_after": cache_after,
                "hits": max(0, raw["llm_calls"] - provider_calls),
                "misses": provider_calls,
                "provider_fallback_calls": adapter_fallback(self.mode, provider_calls),
                "cache_kind": self.freeze_identity.get("cache_kind", "MOCK_PROVIDER"),
            },
            "checkpoint": {"record_level": True, "resumed": False},
            "leakage": {"passed": True, "api_key_value_present": False, "forbidden_fields": []},
            "symbolic_fallback_used": False,
            "oracle_fallback_used": False,
            "mock_fallback_used": False,
            "main_state_pollution": bool(
                raw["probe_result"] and not raw["probe_result"].get("state_unchanged")
            ),
            "patch_registry_state": raw["patch_registry_state"],
        }
        self._assert_public_record(record)
        return record

    def _failure_record(
        self, plan: FocusedRecordPlan, classification: str, stopped_stage: str,
        exc: Exception, cache_before: int, provider_calls: int,
        ledger_before: dict[str, Any], event_before: int,
    ) -> dict[str, Any]:
        self.real_provider_instances += 1 if self.mode == "real" and provider_calls else 0
        self.network_calls += provider_calls if self.mode == "real" and self.provider_factory is None else 0
        ledger_after = self._read_ledger()
        events = self.costs.events[event_before:] if self.costs else []
        provider_errors = [self._public_provider_error(exc)] if isinstance(exc, ProviderError) else []
        record = {
            "schema_version": "focused-live-healing-record-v1",
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": self._execution_mode(),
            "record_id": plan.identity,
            "record_status": "COMPLETE",
            "provider": plan.provider,
            "model": plan.model_id,
            "provider_kind": self.provider_kind,
            "attempt_id": self.attempt_id,
            "method": plan.method,
            "family": plan.family,
            "scenario_id": plan.scenario_id,
            "public_scenario_id": f"public-{plan.family}",
            "source_snapshot": self.source_hash,
            "frozen_fingerprints": deepcopy(self.config.raw["fingerprints"]),
            "stage_trace": self._empty_stage_trace(),
            "live_evidence_trace": {"trace_id": f"unavailable-{plan.identity}", "events": []},
            "evidence_provenance": {"source": "none", "event_count": 0, "phase7_static_reconstruction": False},
            "llm_stage_calls": [],
            "llm_calls": 0,
            "tool_calls": 0,
            "probe_calls": 0,
            "patch_proposal_calls": 0,
            "patch_revision": None,
            "deterministic_gates": {},
            "immediate_repair": None,
            "future_transfer": None,
            "task_success": False,
            "termination_reason": stopped_stage,
            "stopped_stage": stopped_stage,
            "infrastructure_or_method_failure": classification,
            "input_tokens": self._delta(ledger_before, ledger_after, "provider_reported_input_tokens"),
            "output_tokens": self._delta(ledger_before, ledger_after, "provider_reported_output_tokens"),
            "latency_ms": 0.0,
            "api_attempts_delta": self._delta(ledger_before, ledger_after, "api_attempts"),
            "cost_cny_delta": float(ledger_after["spent_cny"]) - float(ledger_before["spent_cny"]),
            "provider_boundary_calls": provider_calls,
            "provider_usage": deepcopy(self.costs.events[event_before:] if self.costs else []),
            "provider_errors": provider_errors,
            "actual_network_attempts": self._delta(ledger_before, ledger_after, "api_attempts"),
            "ledger_before": self._public_ledger(ledger_before),
            "ledger_after": self._public_ledger(ledger_after),
            "cache": {
                "namespace": self.cache_store.namespace,
                "identity_hash": self.cache_identity_hash,
                "entries_before": cache_before,
                "entries_after": self.cache_store.raw_count(),
                "hits": 0,
                "misses": provider_calls,
                "provider_fallback_calls": 0,
                "cache_kind": self.freeze_identity.get("cache_kind"),
            },
            "checkpoint": {"record_level": True, "resumed": False},
            "leakage": {"passed": True, "api_key_value_present": False, "forbidden_fields": []},
            "symbolic_fallback_used": False,
            "oracle_fallback_used": False,
            "mock_fallback_used": False,
            "main_state_pollution": False,
            "patch_registry_state": {"accepted_count": 0, "scope_key": f"{plan.identity}:failed"},
            "error": (
                provider_errors[0] if provider_errors else {
                    "type": type(exc).__name__,
                    "sanitized_message": sanitize_provider_text(exc),
                }
            ),
        }
        self._assert_public_record(record)
        return record

    def _deepseek_gate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        deepseek = [item for item in records if item["provider"] == "deepseek"]
        current_source = (
            _focused_source_snapshot() if self.mode in {"real", "replay-real"}
            else source_snapshot()["source_snapshot_hash"]
        )
        baseline = self._existing_attempt_baseline() or self.run_ledger_before
        incremental_cost = float(self._read_ledger()["spent_cny"]) - float(baseline["spent_cny"])
        checks = {
            "four_records": len(deepseek) == 4,
            "infrastructure_errors_below_two": sum(
                item["infrastructure_or_method_failure"] == "INFRASTRUCTURE_ERROR" for item in deepseek
            ) < 2,
            "no_leakage": all(item["leakage"]["passed"] for item in deepseek),
            "no_fallback": all(
                not item["symbolic_fallback_used"]
                and not item["oracle_fallback_used"]
                and not item["mock_fallback_used"]
                for item in deepseek
            ),
            "no_main_state_pollution": all(not item["main_state_pollution"] for item in deepseek),
            "schemas_valid": all(item["record_status"] == "COMPLETE" for item in deepseek),
            "frozen_source_unchanged": current_source == self.source_hash,
            "soft_cost_gate_not_reached": incremental_cost < float(
                self.config.raw["execution"]["attempt_soft_increment_cny"]
            ),
            "hard_cost_gate_not_reached": incremental_cost < float(
                self.config.raw["execution"]["attempt_hard_increment_cny"]
            ),
        }
        return {
            "provider": "deepseek",
            "records": len(deepseek),
            "passed": all(checks.values()),
            "checks": checks,
            "infrastructure_errors": sum(
                item["infrastructure_or_method_failure"] == "INFRASTRUCTURE_ERROR" for item in deepseek
            ),
            "incremental_cost_cny": incremental_cost,
            "qwen_authorized": all(checks.values()),
        }

    def _manifest(self, ledger: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.output / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("config_hash") != self.config.config_hash
                or existing.get("source_snapshot") != self.source_hash
                or existing.get("attempt_id") != self.attempt_id
            ):
                raise ValueError("existing real manifest does not match this attempt")
            return existing
        return {
            "schema_version": "focused-live-healing-manifest-v1",
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": self._execution_mode(),
            "real_api_status": "RUNNING" if self.mode == "real" and self.provider_kind == "REAL_PROVIDER" else "NOT RUN",
            "config_hash": self.config.config_hash,
            "source_snapshot": self.source_hash,
            "frozen_fingerprints": deepcopy(self.config.raw["fingerprints"]),
            "authorized_record_plan": self.config.plan_dicts(),
            "records_planned": 8,
            "heldout48_overlap": 0,
            "cache_namespace": self.cache_store.namespace,
            "result_namespace": self.config.raw["execution"]["result_namespace"],
            "ledger_baseline": {
                key: (ledger or self.run_ledger_before)[key] for key in LEDGER_BASELINE
            },
            "attempt_id": self.attempt_id,
            "provider_kind": self.provider_kind if self.mode == "real" else "CACHE_ONLY" if "replay" in self.mode else "MOCK_PROVIDER",
            "git": deepcopy(self.git_state),
            "provider_order": ["deepseek", "dashscope"],
            "checkpoint_identity": self._checkpoint_identity(),
            "cost_limits": {
                "incremental_soft_cny": 3.0,
                "incremental_hard_cny": 5.0,
                "cumulative_hard_cny": 50.0,
            },
        }

    def _summary(self, records: list[dict[str, Any]], status: str) -> dict[str, Any]:
        ledger_after = self._read_ledger()
        return {
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": self._execution_mode(),
            "provider_kind": self.provider_kind,
            "attempt_id": self.attempt_id,
            "run_status": status,
            "records_planned": 8,
            "records_completed": len(records),
            "records_resumed": self.resumed_records,
            "accepted": sum(item["termination_reason"] == "LIVE_HEALING_COMPLETE" for item in records),
            "infrastructure_errors": sum(
                item["infrastructure_or_method_failure"] == "INFRASTRUCTURE_ERROR" for item in records
            ),
            "provider_fallback_calls": sum(item["cache"]["provider_fallback_calls"] for item in records),
            "real_provider_instances": self.real_provider_instances if self.mode == "real" else 0,
            "network_calls": self.network_calls,
            "api_attempts_delta": self._delta(self.run_ledger_before, ledger_after, "api_attempts"),
            "input_tokens_delta": self._delta(
                self.run_ledger_before, ledger_after, "provider_reported_input_tokens"
            ),
            "output_tokens_delta": self._delta(
                self.run_ledger_before, ledger_after, "provider_reported_output_tokens"
            ),
            "cost_cny_delta": float(ledger_after["spent_cny"]) - float(self.run_ledger_before["spent_cny"]),
            "symbolic_fallback_count": sum(item["symbolic_fallback_used"] for item in records),
            "oracle_fallback_count": sum(item["oracle_fallback_used"] for item in records),
            "mock_fallback_count": sum(item["mock_fallback_used"] for item in records),
            "ledger_before_sha256": self.run_ledger_before_hash,
            "ledger_before": self._public_ledger(self.run_ledger_before),
            "ledger_after": self._public_ledger(ledger_after),
            "real_api_status": "NOT RUN" if self.provider_kind == "FAKE_PROVIDER_TEST" or self.mode != "real" else (
                "COMPLETED" if status == "COMPLETE" else "PARTIAL"
            ),
        }

    def _bind_checkpoint(self, resume: bool) -> None:
        identity = self._checkpoint_identity()
        if self.checkpoint_identity_path.exists():
            existing = json.loads(self.checkpoint_identity_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise ValueError("config or frozen fingerprint changed; refusing resume")
            if not resume and self.checkpoints.all_records():
                raise FileExistsError("Focused checkpoints exist; pass --resume")
        else:
            if resume and self.checkpoints.all_records():
                raise ValueError("checkpoint records exist without a frozen identity")
            _atomic_json(self.checkpoint_identity_path, identity)

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.config_hash,
            "source_snapshot": self.source_hash,
            "frozen_identity_hash": self.cache_identity_hash,
            "execution_mode": self.mode,
            "attempt_id": self.attempt_id,
            "provider_kind": self.provider_kind,
        }

    def _verify_resumed_record(self, plan: FocusedRecordPlan, record: dict[str, Any]) -> None:
        if record.get("record_id") != plan.identity:
            raise ValueError("checkpoint record identity mismatch")
        if record.get("source_snapshot") != self.source_hash:
            raise ValueError("source snapshot changed; refusing resume")
        if record.get("record_status") != "COMPLETE":
            raise ValueError("incomplete record checkpoint cannot be treated as complete")

    def _validate_real_authorization(
        self, credential_checker: Callable[[], Mapping[str, str]],
    ) -> None:
        if not self.config.run_authorized:
            raise PermissionError("real Focused execution requires run_authorized:true")
        if not self.allow_real_api:
            raise PermissionError("real Focused execution requires --allow-real-api")
        if self.confirmation != CONFIRMATION:
            raise PermissionError(f"real Focused execution requires confirmation {CONFIRMATION}")
        statuses = dict(credential_checker())
        required = {model.api_key_env for model in self.config.models}
        if any(statuses.get(str(name)) != "configured" for name in required):
            raise PermissionError("both Focused Provider credentials must be configured")

    def _validate_attempt_id(self) -> None:
        if not self.attempt_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.attempt_id):
            raise ValueError("real and replay-real require a safe --attempt-id")

    def _validate_real_paths(self, namespace: str) -> None:
        manifest = self.output / "manifest.json"
        if self.output.exists() and any(self.output.iterdir()) and not manifest.exists():
            raise ValueError("real result directory is non-empty and not bound to this attempt")
        cache_directory = self.cache_root / namespace
        identity = cache_directory / "focused_cache_identity.json"
        if cache_directory.exists() and any(cache_directory.iterdir()) and not identity.exists():
            raise ValueError("real Cache directory is non-empty and lacks an attempt identity")

    def _validate_ledger_preflight(self) -> None:
        ledger = self._read_ledger()
        for key, baseline in LEDGER_BASELINE.items():
            actual = ledger.get(key)
            if isinstance(baseline, float):
                if float(actual) + 1e-9 < baseline:
                    raise ValueError(f"formal ledger regressed below baseline: {key}")
            elif int(actual) < baseline:
                raise ValueError(f"formal ledger regressed below baseline: {key}")
        if float(ledger["spent_cny"]) >= 50.0:
            raise CostHardLimit("cumulative hard limit already reached")
        if self.mode == "real" and not (self.output / "manifest.json").exists():
            for key, baseline in LEDGER_BASELINE.items():
                actual = ledger[key]
                if isinstance(baseline, float):
                    if abs(float(actual) - baseline) > 1e-9:
                        raise ValueError(f"new real attempt must start at the frozen ledger baseline: {key}")
                elif actual != baseline:
                    raise ValueError(f"new real attempt must start at the frozen ledger baseline: {key}")

    def _freeze_identity(self, namespace: str) -> dict[str, Any]:
        value = {
            "schema_version": "focused-cache-identity-v1",
            "experiment_kind": self.config.experiment_kind,
            "config_hash": self.config.config_hash,
            "source_snapshot": self.source_hash,
            "tool_catalog_hash": self.config.raw["fingerprints"]["tool_catalog_fingerprint"],
            "policy_capabilities_hash": hashlib.sha256(json.dumps(
                self.config.raw["fingerprints"]["policy_capabilities"], sort_keys=True,
            ).encode()).hexdigest(),
            "frozen_fingerprints_hash": hashlib.sha256(json.dumps(
                self.config.raw["fingerprints"], sort_keys=True,
            ).encode()).hexdigest(),
        }
        if self.mode in {"real", "replay-real"}:
            value.update({
                "cache_kind": self.provider_kind,
                "attempt_id": self.attempt_id,
                "namespace": namespace,
            })
        return value

    def _assert_public_record(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True)
        lowered = encoded.lower()
        forbidden = (
            "source_drift_id", "expected_patch", "evaluator_view",
            "runtime_contract", '"api_key":', '"authorization":', "ground_truth",
        )
        if any(term in lowered for term in forbidden):
            raise ValueError("Focused result contains forbidden hidden or secret material")
        if any(secret in encoded for secret in self._secret_values):
            raise ValueError("Focused result contains an API key value")

    @staticmethod
    def _controller_provider_errors(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            deepcopy(item["provider_error"])
            for item in raw.get("controller_runs", [])
            if isinstance(item.get("provider_error"), Mapping)
        ]

    @staticmethod
    def _public_provider_error(exc: ProviderError) -> dict[str, Any]:
        return exc.public_dict()

    def _read_ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            raise FileNotFoundError("formal Phase 10 ledger is required")
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _existing_attempt_baseline(self) -> Mapping[str, Any] | None:
        path = self.output / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("ledger_baseline")

    def _ledger_snapshot(self) -> tuple[str, dict[str, Any]]:
        return _sha256_bytes(self.ledger_path), self._read_ledger()

    def _execution_mode(self) -> str:
        if self.mode == "mock":
            return "MOCK_INFRASTRUCTURE_VALIDATION"
        if self.mode == "replay":
            return "CACHE_REPLAY"
        if self.mode == "replay-real":
            return "REAL_CACHE_REPLAY"
        if self.provider_kind == "FAKE_PROVIDER_TEST":
            return "REAL_PATH_FAKE_PROVIDER_VALIDATION"
        return "REAL_FOCUSED_CANARY"

    @staticmethod
    def _delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int:
        return int(after[key]) - int(before[key])

    @staticmethod
    def _public_ledger(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "api_attempts": int(value["api_attempts"]),
            "provider_reported_input_tokens": int(value["provider_reported_input_tokens"]),
            "provider_reported_output_tokens": int(value["provider_reported_output_tokens"]),
            "spent_cny": float(value["spent_cny"]),
            "reserved_cny": float(value.get("reserved_cny", 0.0)),
        }

    @staticmethod
    def _last_stage(raw: dict[str, Any]) -> str:
        transitions = raw.get("state_transitions") or []
        return transitions[-1]["new_state"] if transitions else "NOT_STARTED"

    @classmethod
    def _empty_stage_trace(cls) -> list[dict[str, Any]]:
        return [
            {"ordinal": index, "stage": stage, "status": "NOT_REACHED"}
            for index, stage in enumerate(cls._stage_names(), 1)
        ]

    @classmethod
    def _stage_trace(cls, raw: dict[str, Any]) -> list[dict[str, Any]]:
        reached = {item["new_state"] for item in raw.get("state_transitions", [])}
        if "IMMEDIATE_REPAIR" in reached:
            reached.add("REGRESSION_SAFETY_MINIMALITY")
        return [
            {"ordinal": index, "stage": stage, "status": "COMPLETE" if stage in reached else "NOT_REACHED"}
            for index, stage in enumerate(cls._stage_names(), 1)
        ]

    @staticmethod
    def _stage_names() -> tuple[str, ...]:
        return (
            "TASK_EXECUTION", "FAILURE_OBSERVED", "PRELIMINARY_ATTRIBUTION",
            "RETRY_OR_EVIDENCE_COLLECTION", "REPRODUCTION_CHECK", "PROBE_SELECTION",
            "PROBE_EXECUTION", "FINAL_ATTRIBUTION", "PATCH_ELIGIBILITY",
            "PATCH_PROPOSAL", "PATCH_VALIDATION", "IMMEDIATE_REPAIR",
            "REGRESSION_SAFETY_MINIMALITY", "PATCH_ACCEPTED", "FUTURE_TRANSFER",
            "FINAL_EVALUATION",
        )


def adapter_fallback(mode: str, provider_calls: int) -> int:
    return provider_calls if mode in {"replay", "replay-real"} else 0
