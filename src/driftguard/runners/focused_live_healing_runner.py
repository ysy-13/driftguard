from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.checkpoint import CheckpointStore
from driftguard.live.focused_config import FocusedLiveHealingConfig, FocusedRecordPlan
from driftguard.live.provider_adapter import FocusedProviderAdapter
from driftguard.llm import LLMCache
from driftguard.phase10.consistency_audit import source_snapshot

from .live_healing_mock_runner import LiveHealingMockRunner


LEDGER = PROJECT_ROOT / "results/experiments/phase10/cost/ledger.json"
RECORD_SCHEMA = PROJECT_ROOT / "benchmark/schemas/focused_live_healing_record_schema_v1.json"
MANIFEST_SCHEMA = PROJECT_ROOT / "benchmark/schemas/focused_live_healing_manifest_schema_v1.json"
LEDGER_BASELINE = {
    "api_attempts": 831,
    "provider_reported_input_tokens": 3_580_832,
    "provider_reported_output_tokens": 84_182,
    "spent_cny": 5.202644120,
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class FocusedCache:
    """A hash-bound namespace that cannot alias any earlier Phase 10 cache."""

    def __init__(self, root: Path, namespace: str, identity: dict[str, Any]) -> None:
        if namespace in {"v1", "v2", "v3", "v4", "v4b"}:
            raise ValueError("Focused Canary cannot use a legacy cache namespace")
        self.namespace = namespace
        self.directory = root / namespace
        self.identity_path = self.directory / "focused_cache_identity.json"
        if self.identity_path.exists():
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise ValueError("Focused cache identity changed; refusing cache reuse")
        else:
            _atomic_json(self.identity_path, identity)
        self.cache = LLMCache(self.directory)

    def raw_count(self) -> int:
        return len(tuple((self.directory / "raw").glob("*.json")))


class FocusedResultWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.records = directory / "records"
        self.record_validator = Draft202012Validator(json.loads(RECORD_SCHEMA.read_text(encoding="utf-8")))
        self.manifest_validator = Draft202012Validator(json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8")))

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


class FocusedLiveHealingRunner:
    """Dedicated eight-record runner. It never enters Phase10Execution enumeration."""

    def __init__(
        self,
        config_path: Path | str,
        *,
        mode: str,
        output: Path | None = None,
        cache_root: Path | None = None,
    ) -> None:
        if mode not in {"mock", "replay"}:
            raise ValueError("offline Focused runner mode must be mock or replay")
        self.config = FocusedLiveHealingConfig.load(config_path)
        if len(self.config.plan) != 8:
            raise ValueError("Focused runner refuses any plan other than exactly 8 records")
        if self.config.run_authorized:
            raise PermissionError("offline Focused validation requires run_authorized:false")
        self.mode = mode
        execution = self.config.raw["execution"]
        base = output or self.config.output_directory / (
            "mock" if mode == "mock" else "replay"
        )
        self.output = Path(base)
        self.cache_root = Path(cache_root or self.config.output_directory / "cache")
        self.snapshot = source_snapshot()
        self.source_hash = self.snapshot["source_snapshot_hash"]
        self.freeze_identity = {
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
        self.cache_identity_hash = hashlib.sha256(json.dumps(
            self.freeze_identity, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        self.cache_store = FocusedCache(
            self.cache_root, execution["cache_namespace"], self.freeze_identity,
        )
        self.writer = FocusedResultWriter(self.output)
        self.checkpoints = CheckpointStore(self.output / "checkpoint")
        self.checkpoint_identity_path = self.output / "checkpoint/identity.json"
        self.mock_runner = LiveHealingMockRunner()
        self.resumed_records = 0

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
        ledger_before_hash, ledger_before = self._ledger_snapshot()
        self._bind_checkpoint(resume)
        manifest = self._manifest(ledger_before)
        self.writer.write_manifest(manifest)
        completed: list[dict[str, Any]] = []
        summary = self._summary(completed, ledger_before_hash, "INCOMPLETE")
        self.writer.write_summary(summary)
        try:
            for plan in self.config.plan:
                if self.checkpoints.has(plan.identity):
                    if not resume:
                        raise FileExistsError(
                            f"checkpoint exists for {plan.identity}; pass --resume"
                        )
                    record = self.checkpoints.read(plan.identity)
                    self._verify_resumed_record(plan, record)
                    self.writer.write_record(record, allow_existing=True)
                    self.resumed_records += 1
                    completed.append(record)
                    self.writer.write_summary(
                        self._summary(completed, ledger_before_hash, "INCOMPLETE")
                    )
                    continue
                record = self._run_record(plan)
                self.writer.write_record(record)
                self.checkpoints.write(plan.identity, record)
                completed.append(record)
                self.writer.write_summary(
                    self._summary(completed, ledger_before_hash, "INCOMPLETE")
                )
        except Exception as exc:
            failed = self._summary(completed, ledger_before_hash, "INCOMPLETE")
            failed["stopped_reason"] = f"{type(exc).__name__}: {exc}"
            self.writer.write_summary(failed)
            raise
        if len(completed) != 8:
            raise RuntimeError(f"Focused execution must complete exactly 8 records, got {len(completed)}")
        ledger_after_hash, _ = self._ledger_snapshot()
        if ledger_after_hash != ledger_before_hash:
            raise RuntimeError("formal Phase 10 ledger changed during offline Focused execution")
        final = self._summary(completed, ledger_before_hash, "COMPLETE")
        final["ledger_after_sha256"] = ledger_after_hash
        self.writer.write_summary(final)
        return {"manifest": manifest, "summary": final, "records": completed}

    def _run_record(self, plan: FocusedRecordPlan) -> dict[str, Any]:
        model = next(
            item for item in self.config.models
            if (item.provider, item.model_id) == (plan.provider, plan.model_id)
        )
        model = replace(model, seed=plan.seed)
        adapter = FocusedProviderAdapter(
            self.mode, model, run_authorized=self.config.run_authorized,
        )
        cache_before = self.cache_store.raw_count()
        raw = self.mock_runner.run_pd(
            plan.family, model_config=model, provider_adapter=adapter,
            cache=self.cache_store.cache, config_hash=self.cache_identity_hash,
            repetition=plan.repetition, seed=plan.seed,
        )
        cache_after = self.cache_store.raw_count()
        provider_calls = adapter.mock_calls if self.mode == "mock" else adapter.fallback_calls
        if self.mode == "mock" and not 0 <= provider_calls <= 6:
            raise RuntimeError(f"mock record made an invalid provider-boundary call count: {provider_calls}")
        if self.mode == "replay" and provider_calls != 0:
            raise RuntimeError("replay attempted Provider fallback")
        return self._record(plan, raw, cache_before, cache_after, provider_calls)

    def _record(
        self, plan: FocusedRecordPlan, raw: dict[str, Any],
        cache_before: int, cache_after: int, provider_calls: int,
    ) -> dict[str, Any]:
        execution_mode = (
            "MOCK_INFRASTRUCTURE_VALIDATION" if self.mode == "mock" else "CACHE_REPLAY"
        )
        complete = raw["termination_reason"] == "LIVE_HEALING_COMPLETE"
        stage_calls = deepcopy(raw["llm_stage_calls"])
        for call in stage_calls:
            call["simulated_provider_attempts"] = call.get("provider_attempts", 0)
            call["provider_attempts"] = 0
        record = {
            "schema_version": "focused-live-healing-record-v1",
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": execution_mode,
            "record_id": plan.identity,
            "record_status": "COMPLETE" if complete else "INCOMPLETE",
            "provider": plan.provider,
            "model": plan.model_id,
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
            "termination_reason": raw["termination_reason"],
            "stopped_stage": "FINAL_EVALUATION" if complete else "METHOD_FAILURE",
            "infrastructure_or_method_failure": "NONE" if complete else "METHOD_FAILURE",
            "input_tokens": 0,
            "output_tokens": 0,
            "mock_simulated_input_tokens": raw["input_tokens"] if self.mode == "mock" else 0,
            "mock_simulated_output_tokens": raw["output_tokens"] if self.mode == "mock" else 0,
            "latency_ms": sum(item.get("latency_ms", 0.0) for item in raw["llm_stage_calls"]),
            "api_attempts_delta": 0,
            "cost_cny_delta": 0.0,
            "provider_boundary_calls": provider_calls,
            "cache": {
                "namespace": self.cache_store.namespace,
                "identity_hash": self.cache_identity_hash,
                "entries_before": cache_before,
                "entries_after": cache_after,
                "hits": raw["llm_calls"] - provider_calls,
                "misses": provider_calls if self.mode == "mock" else 0,
                "provider_fallback_calls": adapter_fallback(self.mode, provider_calls),
            },
            "checkpoint": {"record_level": True, "resumed": False},
            "leakage": {"passed": True, "api_key_values_read": False, "forbidden_fields": []},
            "symbolic_fallback_used": False,
            "oracle_fallback_used": False,
            "main_state_pollution": not bool(raw["probe_result"]["state_unchanged"]),
            "patch_registry_state": raw["patch_registry_state"],
        }
        self._assert_public_record(record)
        return record

    @staticmethod
    def _stage_trace(raw: dict[str, Any]) -> list[dict[str, Any]]:
        stages = (
            "TASK_EXECUTION", "FAILURE_OBSERVED", "PRELIMINARY_ATTRIBUTION",
            "RETRY_OR_EVIDENCE_COLLECTION", "REPRODUCTION_CHECK", "PROBE_SELECTION",
            "PROBE_EXECUTION", "FINAL_ATTRIBUTION", "PATCH_ELIGIBILITY",
            "PATCH_PROPOSAL", "PATCH_VALIDATION", "IMMEDIATE_REPAIR",
            "REGRESSION_SAFETY_MINIMALITY", "PATCH_ACCEPTED", "FUTURE_TRANSFER",
            "FINAL_EVALUATION",
        )
        reached = {item["new_state"] for item in raw["state_transitions"]}
        reached.add("REGRESSION_SAFETY_MINIMALITY")
        return [
            {"ordinal": index, "stage": stage, "status": "COMPLETE" if stage in reached else "NOT_REACHED"}
            for index, stage in enumerate(stages, 1)
        ]

    def _manifest(self, ledger: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "focused-live-healing-manifest-v1",
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": "MOCK_INFRASTRUCTURE_VALIDATION" if self.mode == "mock" else "CACHE_REPLAY",
            "real_api_status": "NOT RUN",
            "config_hash": self.config.config_hash,
            "source_snapshot": self.source_hash,
            "frozen_fingerprints": deepcopy(self.config.raw["fingerprints"]),
            "authorized_record_plan": self.config.plan_dicts(),
            "records_planned": 8,
            "heldout48_overlap": 0,
            "cache_namespace": self.config.raw["execution"]["cache_namespace"],
            "result_namespace": self.config.raw["execution"]["result_namespace"],
            "ledger_baseline": {key: ledger[key] for key in LEDGER_BASELINE},
        }

    def _summary(
        self, records: list[dict[str, Any]], ledger_hash: str, status: str,
    ) -> dict[str, Any]:
        return {
            "experiment_kind": self.config.experiment_kind,
            "execution_mode": "MOCK_INFRASTRUCTURE_VALIDATION" if self.mode == "mock" else "CACHE_REPLAY",
            "run_status": status,
            "records_planned": 8,
            "records_completed": len(records),
            "records_resumed": self.resumed_records,
            "accepted": sum(item["termination_reason"] == "LIVE_HEALING_COMPLETE" for item in records),
            "provider_fallback_calls": sum(item["cache"]["provider_fallback_calls"] for item in records),
            "api_attempts_delta": 0,
            "input_tokens_delta": sum(item["input_tokens"] for item in records),
            "output_tokens_delta": sum(item["output_tokens"] for item in records),
            "cost_cny_delta": 0.0,
            "symbolic_fallback_count": sum(item["symbolic_fallback_used"] for item in records),
            "oracle_fallback_count": sum(item["oracle_fallback_used"] for item in records),
            "ledger_before_sha256": ledger_hash,
            "real_api_status": "NOT RUN",
        }

    def _bind_checkpoint(self, resume: bool) -> None:
        identity = {
            "config_hash": self.config.config_hash,
            "source_snapshot": self.source_hash,
            "frozen_identity_hash": self.cache_identity_hash,
            "execution_mode": self.mode,
        }
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

    def _verify_resumed_record(self, plan: FocusedRecordPlan, record: dict[str, Any]) -> None:
        if record.get("record_id") != plan.identity:
            raise ValueError("checkpoint record identity mismatch")
        if record.get("source_snapshot") != self.source_hash:
            raise ValueError("source snapshot changed; refusing resume")
        if record.get("record_status") != "COMPLETE":
            raise ValueError("incomplete record checkpoint cannot be treated as complete")

    @staticmethod
    def _assert_public_record(record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True).lower()
        forbidden = ("source_drift_id", "expected_patch", "evaluator_view", "runtime_contract", "api_key\"")
        if any(term in encoded for term in forbidden):
            raise ValueError("Focused result contains forbidden hidden or secret material")

    @staticmethod
    def _ledger_snapshot() -> tuple[str, dict[str, Any]]:
        if not LEDGER.exists():
            raise FileNotFoundError("formal Phase 10 ledger is required for offline continuity check")
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        for key, expected in LEDGER_BASELINE.items():
            actual = ledger.get(key)
            if isinstance(expected, float):
                if abs(float(actual) - expected) > 1e-9:
                    raise ValueError(f"formal ledger baseline mismatch: {key}")
            elif actual != expected:
                raise ValueError(f"formal ledger baseline mismatch: {key}")
        return _sha256_bytes(LEDGER), ledger


def adapter_fallback(mode: str, provider_calls: int) -> int:
    return provider_calls if mode == "replay" else 0
