from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
import yaml

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.evidence.models import thaw
from driftguard.llm import (
    InvalidStructuredOutput, LLMCache, LLMProvider, ModelConfig,
    OpenAICompatibleProvider, ProviderCapabilityAdapter, ProviderError,
    ProviderRequest, ProviderResponse, RequestRateLimiter, model_api_key,
    parse_structured_output,
)
from driftguard.llm.redaction import assert_secret_absent
from driftguard.phase10.pricing import PricingCatalog
from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner
from driftguard.runners.focused_live_healing_runner import (
    AtomicFocusedLedger, LEDGER, _atomic_json, _ledger_dominates,
    _public_ledger_snapshot, _sha256_bytes, _validated_ledger,
)

from .protocol import (
    BenchmarkErrorClass, CanonicalToolRegistry, EvidenceViewBuilder,
    OutputTruncated, TrueEvidenceLeakage,
    classify_error, expected_for_evaluator, normalize_prediction,
)
from .runner import OUTPUT_SCHEMA_PATH, PRICING_PATH, PROMPT_PATH, VIEW_DEFINITION_PATH
from .heldout_gate import InfrastructureGateV2


EXPERIMENT_MODE = "SPECDRIFTBENCH_COMPONENT_HELDOUT_432"
CONFIRM_432 = "RUN-EXACTLY-432"
CONFIRM_432_V2 = "RUN-EXACTLY-432-V2"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/specdriftbench_component_heldout432_v1.yaml"
CONFIG_V2_PATH = PROJECT_ROOT / "configs/experiments/specdriftbench_component_heldout432_v2.yaml"
CANARY_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/specdriftbench_component_canary_v1_moonshot_cn_k26_nonthinking_v1.yaml"
ANALYSIS_PLAN_PATH = PROJECT_ROOT / "benchmark/analysis/specdriftbench_heldout432_analysis_plan_v1.json"
RECORD_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/specdriftbench_component_heldout_record_schema_v1.json"
MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/specdriftbench_component_heldout_manifest_schema_v1.json"
SUMMARY_SCHEMA_PATH = PROJECT_ROOT / "benchmark/schemas/specdriftbench_component_heldout_summary_schema_v1.json"
FORMAL_ATTEMPT_ROOT = PROJECT_ROOT / "results/experiments/phase11/specdriftbench_component_heldout432/attempts"

DEVELOPMENT_FAMILIES = ("M01", "M06", "M11", "M16")
HELDOUT_FAMILIES = (
    "M02", "M03", "M04", "M05", "M07", "M08", "M09", "M10",
    "M12", "M13", "M14", "M15", "M17", "M18", "M19", "M20",
)
VARIANTS = ("AE", "TF", "PD")
EVIDENCE_VIEWS = ("FIRST_FAILURE", "RETRY_HISTORY", "FULL_EVIDENCE")
PROVIDER_ORDER = ("deepseek", "dashscope", "moonshot")
REAL_CACHE_NAMESPACE = "specdriftbench_component_heldout432_real_v1"
FAKE_CACHE_NAMESPACE = "specdriftbench_component_heldout432_fake_validation_v1"
REAL_CACHE_NAMESPACE_V2 = "specdriftbench_component_heldout432_real_v2"
FAKE_CACHE_NAMESPACE_V2 = "specdriftbench_component_heldout432_fake_validation_v2"
EXCLUDED_V1_ATTEMPT = "specdriftbench-heldout432-20260718-af3db03-01"

FROZEN_FINGERPRINTS = {
    "prompt_sha256": "d499cdb339972ad5a9424339519f4c64a9cb9096afcddd46b6309ccd7cc2a47e",
    "attribution_schema_sha256": "fb1a66c6d331fc1af98ef6985518f19e08ef9733f2ad07624dbfd962ce44b6f9",
    "evidence_view_sha256": "94376b30b8554539602922f6da656dedcfc7a376650daec64735658af39585ac",
    "pricing_sha256": "5ba33e1c6769dd208ab43241504dce0c161d4bbd2d92b3db580d26ea41868351",
    "tool_registry_sha256": "5be5dacb739ed05cb81b5a005de0faf414312d9bb9ef0664eafe8a1000ed7205",
}

_HIDDEN_PROVIDER_KEYS = {
    "expected_class", "expected_category", "expected_target", "expected_location",
    "family", "family_id", "matched_case_id", "variant", "variant_code",
    "ground_truth_label", "evaluator_metadata", "future_episode_evidence",
    "source_drift_id", "expected_patch", "expected_patch_ref",
    "drift_id", "runtime_contract", "runtime_profile", "expected_action",
    "persistent_patch_allowed", "development_prediction", "development_result",
    "canary_prediction", "canary_predictions", "canary_result", "canary_results",
}
_HIDDEN_PROVIDER_TEXT = (
    "canary prediction", "canary result", "expected class", "expected category",
    "expected target", "expected location", "cross-scenario history",
    "variant code", "family id", "drift id", "future episode",
)
_SEMANTIC_REPLAY_FIELDS = (
    "record_id", "attempt_id", "experiment_mode", "execution_status", "provider", "model",
    "provider_profile_id", "max_tokens", "family", "variant", "evidence_view", "repetition",
    "paired_group_id", "public_scenario_id", "evidence_sha256", "raw_response_ref",
    "raw_response_sha256", "parsed_result", "normalized_prediction", "schema_valid",
    "format_repairs", "error_class", "error", "evaluation", "input_tokens", "output_tokens",
    "reasoning_tokens", "latency_ms", "provider_attempts", "finish_reason", "content_length",
    "fallback_used", "main_state_pollution", "leakage", "termination_reason",
)


class HeldoutProtocolError(ValueError):
    pass


class HeldoutCacheMiss(RuntimeError):
    pass


@dataclass(frozen=True)
class HeldoutPlan:
    ordinal: int
    provider: str
    model: str
    family: str
    variant: str
    evidence_view: str
    repetition: int = 1

    @property
    def paired_group_id(self) -> str:
        material = f"heldout432-pair-v1|{self.provider}|{self.family}|{self.variant}"
        return "pair-" + hashlib.sha256(material.encode()).hexdigest()[:16]

    @property
    def record_id(self) -> str:
        material = (
            f"heldout432-record-v1|{self.provider}|{self.model}|{self.family}|"
            f"{self.variant}|{self.evidence_view}|{self.repetition}"
        )
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal, "record_id": self.record_id,
            "paired_group_id": self.paired_group_id, "provider": self.provider,
            "model": self.model, "family": self.family, "variant": self.variant,
            "evidence_view": self.evidence_view, "repetition": self.repetition,
        }


def assert_heldout_provider_visible(value: Any, path: str = "$") -> None:
    """Reject evaluator-only material while allowing the canonical displayed API spec.

    The displayed specification legitimately names public request fields such as
    ``verification_token``.  The older Canary guard treats that phrase as hidden,
    so the held-out boundary uses the stricter task-specific denylist instead of
    weakening or changing the frozen canonical specification.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _HIDDEN_PROVIDER_KEYS:
                raise TrueEvidenceLeakage(f"held-out evaluator field at {path}.{key}")
            assert_heldout_provider_visible(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_heldout_provider_visible(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _HIDDEN_PROVIDER_TEXT):
            raise TrueEvidenceLeakage(f"held-out hidden content at {path}")
        if any(marker in lowered for marker in (
            "ground_truth_label", "evaluator_metadata", "source_drift_id",
            "expected_patch", "future_episode_evidence", "hidden runtime contract",
        )):
            raise TrueEvidenceLeakage(f"held-out hidden content at {path}")
        if re.search(r"\b(?:M\d{2}|(?:ICD|RSD|WPD|SED)-\d{2})\b", value, re.IGNORECASE):
            raise TrueEvidenceLeakage(f"held-out family/source content at {path}")


class HeldoutEvidenceViewBuilder(EvidenceViewBuilder):
    """Uses the frozen Evidence View construction over only the 16 held-out families."""

    def __init__(self) -> None:
        self.runner = AttributionConformanceRunner(strict_leakage_check=False)
        self.families = {
            family["matched_case_id"]: family for family in self.runner.matched["families"]
            if family["matched_case_id"] in HELDOUT_FAMILIES
        }
        if tuple(sorted(self.families)) != tuple(sorted(HELDOUT_FAMILIES)):
            raise HeldoutProtocolError("canonical matched failures do not contain the exact held-out family set")

    def build(self, family_id: str, variant: str, view_name: str | None = None) -> dict[str, Any]:
        if family_id not in HELDOUT_FAMILIES or variant not in VARIANTS:
            raise HeldoutProtocolError("Evidence request is outside the frozen held-out split")
        if view_name is not None and view_name not in EVIDENCE_VIEWS:
            raise HeldoutProtocolError("unknown held-out Evidence View")
        family = self.families[family_id]
        scenario = next(item for item in family["scenarios"] if item["variant_code"] == variant)
        artifacts = self.runner._run_scenario(family, scenario, return_agent_artifacts=True)
        trace = artifacts["agent_view"].trace.to_dict()
        selected = view_name or EVIDENCE_VIEWS[0]
        events = [deepcopy(event) for event in trace["events"] if self._include(event, selected)]
        if selected == "FULL_EVIDENCE":
            events = self._ensure_full_view_is_strict(events)
        evidence = {
            "schema_version": "specdriftbench-agent-view-v1",
            "evidence_view": selected,
            "public_scenario_id": artifacts["public_scenario_id"],
            "user_intent": scenario["agent_visible"]["evidence_bundle"]["user_task"],
            "displayed_specification": thaw(artifacts["agent_view"].displayed_spec),
            "evidence_trace": {
                **{key: deepcopy(child) for key, child in trace.items() if key != "events"},
                "events": events,
            },
        }
        assert_heldout_provider_visible(evidence)
        evidence["evidence_sha256"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=list).encode()
        ).hexdigest()
        return evidence

    @staticmethod
    def _ensure_full_view_is_strict(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Materialize the frozen FULL response-shape observation when no probe exists.

        Transient-failure traces can end after a successful retry.  The frozen
        Evidence View definition still includes a response-shape observation in
        FULL_EVIDENCE.  This event is a deterministic projection of that already
        Agent-visible retry response; it never reads evaluator state.
        """
        if any(event["event_type"] not in EvidenceViewBuilder.RETRY_TYPES for event in events):
            return events
        retry = next((event for event in reversed(events) if event["event_type"] == "retry_result"), None)
        if retry is None:
            raise HeldoutProtocolError("FULL_EVIDENCE has neither a safe probe nor a retry response")
        response = retry.get("visible_runtime_response") or {}
        payload = response.get("payload") if isinstance(response, Mapping) else None
        payload_keys = sorted(str(key) for key in payload) if isinstance(payload, Mapping) else []
        observation = {
            "event_id": f"{retry['trace_id']}-FULL-SHAPE",
            "trace_id": retry["trace_id"],
            "public_scenario_id": retry["public_scenario_id"],
            "episode_number": retry["episode_number"],
            "sequence_number": max(int(event["sequence_number"]) for event in events) + 1,
            "event_type": "response_shape_observation",
            "tool_id": retry.get("tool_id"),
            "displayed_request": {}, "local_validation_result": {},
            "visible_runtime_response": {},
            "normalized_observation": {
                "source_event_id": retry["event_id"],
                "http_status": response.get("status_code") if isinstance(response, Mapping) else None,
                "response_success": (retry.get("normalized_observation") or {}).get("response_success"),
                "top_level_payload_keys": payload_keys,
            },
            "visible_state_diff": [],
            "before_state_hash": retry.get("before_state_hash"),
            "after_state_hash": retry.get("after_state_hash"),
            "displayed_spec_fingerprint": retry.get("displayed_spec_fingerprint"),
            "historical_evidence_refs": [],
            "probe_metadata": {"derived_from_agent_visible_retry": True},
            "timestamp": retry.get("timestamp"),
        }
        return [*events, observation]


def validate_paired_evidence(evidence_by_view: Mapping[str, Mapping[str, Any]]) -> None:
    if set(evidence_by_view) != set(EVIDENCE_VIEWS):
        raise HeldoutProtocolError("paired evidence must contain exactly the three frozen Evidence Views")
    first, retry, full = (evidence_by_view[name] for name in EVIDENCE_VIEWS)
    for name, evidence in evidence_by_view.items():
        if evidence.get("evidence_view") != name:
            raise HeldoutProtocolError("Evidence View label does not match its paired slot")
        assert_heldout_provider_visible(evidence)
    stable_fields = ("public_scenario_id", "user_intent", "displayed_specification")
    if any(first[field] != retry[field] or retry[field] != full[field] for field in stable_fields):
        raise HeldoutProtocolError("paired Evidence Views changed scenario, task, or displayed specification")
    trace_ids = {
        evidence["evidence_trace"].get("trace_id")
        for evidence in evidence_by_view.values()
    }
    if len(trace_ids) != 1:
        raise HeldoutProtocolError("cross-scenario history detected in paired Evidence Views")
    event_ids = {
        name: {event["event_id"] for event in evidence["evidence_trace"]["events"]}
        for name, evidence in evidence_by_view.items()
    }
    if not (event_ids["FIRST_FAILURE"] < event_ids["RETRY_HISTORY"] < event_ids["FULL_EVIDENCE"]):
        raise HeldoutProtocolError("Evidence Views are not strict monotone subsets")


def validate_heldout_plan(plan: Sequence[HeldoutPlan]) -> dict[str, int]:
    if len(plan) != 432:
        raise HeldoutProtocolError("held-out Component plan must contain exactly 432 records")
    if len({item.record_id for item in plan}) != 432:
        raise HeldoutProtocolError("held-out record IDs are not unique")
    providers = Counter(item.provider for item in plan)
    families = Counter(item.family for item in plan)
    variants = Counter(item.variant for item in plan)
    views = Counter(item.evidence_view for item in plan)
    if providers != {provider: 144 for provider in PROVIDER_ORDER}:
        raise HeldoutProtocolError("each Provider must have exactly 144 records")
    if families != {family: 27 for family in HELDOUT_FAMILIES}:
        raise HeldoutProtocolError("each held-out family must have exactly 27 records")
    if variants != {variant: 144 for variant in VARIANTS}:
        raise HeldoutProtocolError("each variant must have exactly 144 records")
    if views != {view: 144 for view in EVIDENCE_VIEWS}:
        raise HeldoutProtocolError("each Evidence View must have exactly 144 records")
    if set(families) & set(DEVELOPMENT_FAMILIES):
        raise HeldoutProtocolError("development family leaked into held-out plan")
    if set(families) != set(HELDOUT_FAMILIES):
        raise HeldoutProtocolError("held-out family set changed")
    if any(item.repetition != 1 for item in plan):
        raise HeldoutProtocolError("held-out Component repetitions must equal 1")
    for provider in PROVIDER_ORDER:
        provider_rows = [item for item in plan if item.provider == provider]
        if Counter(item.evidence_view for item in provider_rows) != {view: 48 for view in EVIDENCE_VIEWS}:
            raise HeldoutProtocolError("Provider × Evidence View balance changed")
        if Counter(item.variant for item in provider_rows) != {variant: 48 for variant in VARIANTS}:
            raise HeldoutProtocolError("Provider × variant balance changed")
    pairs: dict[str, list[HeldoutPlan]] = defaultdict(list)
    for item in plan:
        pairs[item.paired_group_id].append(item)
    if len(pairs) != 144:
        raise HeldoutProtocolError("held-out plan must contain exactly 144 paired groups")
    for rows in pairs.values():
        if len(rows) != 3 or Counter(item.evidence_view for item in rows) != {view: 1 for view in EVIDENCE_VIEWS}:
            raise HeldoutProtocolError("each paired group must contain each Evidence View exactly once")
        stable = {(item.provider, item.model, item.family, item.variant, item.repetition) for item in rows}
        if len(stable) != 1:
            raise HeldoutProtocolError("paired group changed a non-Evidence-View dimension")
    return {
        "records": 432, "providers": 3, "per_provider": 144,
        "heldout_families": 16, "per_family": 27, "variants": 3,
        "per_variant": 144, "evidence_views": 3, "per_evidence_view": 144,
        "paired_groups": 144, "records_per_paired_group": 3,
    }


def _actual_frozen_fingerprints() -> dict[str, str]:
    return {
        "prompt_sha256": _sha256_bytes(PROMPT_PATH),
        "attribution_schema_sha256": _sha256_bytes(OUTPUT_SCHEMA_PATH),
        "evidence_view_sha256": _sha256_bytes(VIEW_DEFINITION_PATH),
        "pricing_sha256": _sha256_bytes(PRICING_PATH),
        "tool_registry_sha256": CanonicalToolRegistry.from_displayed_spec().fingerprint(),
    }


def validate_frozen_fingerprints(expected: Mapping[str, Any]) -> None:
    if dict(expected) != FROZEN_FINGERPRINTS:
        raise HeldoutProtocolError("held-out config does not contain the frozen fingerprint set")
    actual = _actual_frozen_fingerprints()
    if actual != FROZEN_FINGERPRINTS:
        changed = sorted(key for key in actual if actual[key] != FROZEN_FINGERPRINTS[key])
        raise HeldoutProtocolError("frozen canonical assets changed: " + ", ".join(changed))


@dataclass(frozen=True)
class HeldoutConfig:
    path: Path
    raw: dict[str, Any]
    models: tuple[ModelConfig, ...]
    plan: tuple[HeldoutPlan, ...]
    counts: dict[str, int]
    config_hash: str

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "HeldoutConfig":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise HeldoutProtocolError("held-out config root must be an object")
        required = {
            "experiment_mode", "run_authorized", "expected_records", "repetitions", "heldout_only",
            "development_families", "heldout_families", "variants", "evidence_views", "models",
            "controls", "budget", "execution", "analysis_plan", "frozen_fingerprints",
        }
        is_v2 = raw.get("infrastructure_gate_version") == 2
        if is_v2:
            required |= {"infrastructure_gate_version", "protocol_amendment", "excluded_attempts"}
        if set(raw) != required:
            raise HeldoutProtocolError("held-out config fields changed")
        if is_v2 and (
            raw["protocol_amendment"] != "1.1"
            or raw["excluded_attempts"] != [EXCLUDED_V1_ATTEMPT]
        ):
            raise HeldoutProtocolError("V2 amendment identity or excluded V1 Attempt changed")
        if raw["experiment_mode"] != EXPERIMENT_MODE:
            raise HeldoutProtocolError("wrong held-out experiment mode")
        if not isinstance(raw["run_authorized"], bool):
            raise HeldoutProtocolError("run_authorized must be a boolean")
        if raw["expected_records"] != 432 or raw["repetitions"] != 1 or raw["heldout_only"] is not True:
            raise HeldoutProtocolError("held-out count, repetition, or split gate changed")
        if tuple(raw["development_families"]) != DEVELOPMENT_FAMILIES:
            raise HeldoutProtocolError("development family split changed")
        if tuple(raw["heldout_families"]) != HELDOUT_FAMILIES:
            raise HeldoutProtocolError("held-out family split changed")
        if tuple(raw["variants"]) != VARIANTS or tuple(raw["evidence_views"]) != EVIDENCE_VIEWS:
            raise HeldoutProtocolError("variant or Evidence View set changed")
        canary_raw = yaml.safe_load(CANARY_CONFIG_PATH.read_text(encoding="utf-8"))
        if raw["models"] != canary_raw["models"]:
            raise HeldoutProtocolError("Provider model parameters differ from the frozen Canary")
        models = tuple(ModelConfig.from_mapping(item) for item in raw["models"])
        if tuple(model.provider for model in models) != PROVIDER_ORDER:
            raise HeldoutProtocolError("Provider order changed")
        controls = raw["controls"]
        if set(controls) != {
            "sandbox_enabled", "tool_calling_enabled", "patch_enabled",
            "repair_enabled", "future_transfer_enabled",
        } or any(value is not False for value in controls.values()):
            raise HeldoutProtocolError("held-out execution enables a prohibited agent feature")
        budget = raw["budget"]
        if (
            float(budget["attempt_soft_limit_cny"]) != 15.0
            or float(budget["attempt_hard_limit_cny"]) != 20.0
            or float(budget["global_hard_limit_cny"]) != 50.0
            or int(budget["max_input_tokens_per_record"]) != 60000
            or int(budget["max_output_tokens_per_record"]) != 2048
            or int(budget["max_format_repairs"]) != 1
        ):
            raise HeldoutProtocolError("held-out budget or format-repair limit changed")
        execution = raw["execution"]
        if tuple(execution["provider_order"]) != PROVIDER_ORDER:
            raise HeldoutProtocolError("execution Provider order changed")
        expected_real_cache = REAL_CACHE_NAMESPACE_V2 if is_v2 else REAL_CACHE_NAMESPACE
        expected_fake_cache = FAKE_CACHE_NAMESPACE_V2 if is_v2 else FAKE_CACHE_NAMESPACE
        if execution["cache_namespace"] != expected_real_cache:
            raise HeldoutProtocolError("real held-out Cache namespace changed")
        if execution["fake_cache_namespace"] != expected_fake_cache:
            raise HeldoutProtocolError("FakeProvider Cache namespace changed")
        forbidden = ("canary", "focused", "phase9", "end_to_end", "fake")
        if any(token in execution["cache_namespace"].lower() for token in forbidden):
            raise HeldoutProtocolError("real held-out Cache namespace reuses a prohibited experiment")
        expected_result_namespace = (
            "specdriftbench_component_heldout432_v2" if is_v2
            else "specdriftbench_component_heldout432_v1"
        )
        common_execution_valid = execution["result_namespace"] == expected_result_namespace
        if (
            not common_execution_valid
            or int(execution["parallelism"]) != 1
            or int(execution["checkpoint_interval"]) != 1
            or execution["resume_enabled"] is not True
            or execution["replay_enabled"] is not True
            or float(execution["rate_limit_per_second"]) != 1.0
            or int(execution["seed"]) != 20260718
        ):
            raise HeldoutProtocolError("held-out result, checkpoint, resume, or replay gate changed")
        if is_v2:
            if (
                set(execution) != {
                    "cache_namespace", "fake_cache_namespace", "result_namespace", "provider_order",
                    "parallelism", "rate_limit_per_second", "checkpoint_interval", "resume_enabled",
                    "replay_enabled", "consecutive_final_infrastructure_error_limit",
                    "minimum_records_for_error_rate_gate", "final_infrastructure_error_rate_limit", "seed",
                }
                or int(execution["consecutive_final_infrastructure_error_limit"]) != 3
                or int(execution["minimum_records_for_error_rate_gate"]) != 24
                or float(execution["final_infrastructure_error_rate_limit"]) != 0.10
            ):
                raise HeldoutProtocolError("V2 infrastructure gate thresholds changed")
        elif int(execution["infrastructure_error_limit"]) != 2:
            raise HeldoutProtocolError("V1 infrastructure gate changed")
        validate_frozen_fingerprints(raw["frozen_fingerprints"])
        if raw["analysis_plan"] != "benchmark/analysis/specdriftbench_heldout432_analysis_plan_v1.json":
            raise HeldoutProtocolError("held-out analysis plan path changed")
        analysis = json.loads((PROJECT_ROOT / raw["analysis_plan"]).read_text(encoding="utf-8"))
        if (
            analysis.get("status") != "FROZEN_BEFORE_REAL_RESULTS"
            or analysis.get("bootstrap") != {
                "unit": "family", "repetitions": 10000, "seed": 20260718,
                "confidence_level": 0.95, "method": "family_clustered_bootstrap",
            }
        ):
            raise HeldoutProtocolError("held-out analysis plan is not frozen")
        ordinal = 0
        plan: list[HeldoutPlan] = []
        for model in models:
            for family in HELDOUT_FAMILIES:
                for variant in VARIANTS:
                    for evidence_view in EVIDENCE_VIEWS:
                        ordinal += 1
                        plan.append(HeldoutPlan(
                            ordinal, model.provider, model.model_id, family, variant, evidence_view, 1,
                        ))
        counts = validate_heldout_plan(plan)
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        return cls(source, raw, models, tuple(plan), counts, hashlib.sha256(encoded).hexdigest())

    @property
    def run_authorized(self) -> bool:
        return self.raw["run_authorized"] is True

    @property
    def infrastructure_gate_version(self) -> int:
        return int(self.raw.get("infrastructure_gate_version", 1))

    def model(self, provider: str) -> ModelConfig:
        return next(model for model in self.models if model.provider == provider)


class _HeldoutLedgerConfigAdapter:
    def __init__(self, config: HeldoutConfig):
        self.config_hash = config.config_hash
        budget = config.raw["budget"]
        self.raw = {
            "execution": {
                "attempt_soft_increment_cny": float(budget["attempt_soft_limit_cny"]),
                "attempt_hard_increment_cny": float(budget["attempt_hard_limit_cny"]),
                "total_hard_limit_cny": float(budget["global_hard_limit_cny"]),
            },
            "budget": {
                "full_hard_limit_cny": float(budget["global_hard_limit_cny"]),
                "max_input_tokens_per_record": int(budget["max_input_tokens_per_record"]),
                "max_output_tokens_per_record": int(budget["max_output_tokens_per_record"]),
            },
        }


class HeldoutFakeProvider(LLMProvider):
    """Non-oracle offline protocol boundary; it never reads evaluator truth or performs I/O."""

    def __init__(self, model: ModelConfig):
        self.model = model
        self.calls = 0

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        assert_heldout_provider_visible(request.messages)
        self.calls += 1
        output = {
            "predicted_class": "AGENT_ERROR",
            "drift_category": "NONE",
            "target_tool": None,
            "normalized_location": None,
            "confidence": 0.5,
            "abstain": False,
            "concise_reason": "Offline FakeProvider used only the supplied visible evidence.",
        }
        raw = json.dumps(output, sort_keys=True)
        input_tokens = max(1, sum(len(message["content"]) for message in request.messages) // 4)
        output_tokens = max(1, len(raw) // 4)
        response_id = hashlib.sha256(
            f"fake-heldout|{self.model.provider}|{request.public_scenario_id}|{self.calls}".encode()
        ).hexdigest()[:20]
        return ProviderResponse(
            response_id, self.model.model_id, None, raw,
            input_tokens, output_tokens, input_tokens + output_tokens,
            0.0, 1, "stop",
        )


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "tree": tree, "dirty": bool(status.strip())}


def heldout_source_snapshot(config: HeldoutConfig | None = None) -> str:
    config = config or HeldoutConfig.load()
    config_path = (
        "configs/experiments/specdriftbench_component_heldout432_v2.yaml"
        if config.infrastructure_gate_version == 2
        else "configs/experiments/specdriftbench_component_heldout432_v1.yaml"
    )
    paths = (
        "src/driftguard/specdriftbench/heldout.py",
        "src/driftguard/specdriftbench/heldout_gate.py",
        "src/driftguard/specdriftbench/protocol.py",
        "src/driftguard/llm/provider.py",
        "src/driftguard/llm/provider_capabilities.py",
        config_path,
        "benchmark/analysis/specdriftbench_heldout432_analysis_plan_v1.json",
        "benchmark/schemas/specdriftbench_component_heldout_record_schema_v1.json",
        "benchmark/schemas/specdriftbench_component_heldout_manifest_schema_v1.json",
        "benchmark/schemas/specdriftbench_component_heldout_summary_schema_v1.json",
    )
    value = {path: _sha256_bytes(PROJECT_ROOT / path) for path in paths}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


ProviderFactory = Callable[[ModelConfig], LLMProvider]


class HeldoutRunner:
    MODES = {"fake", "real", "replay"}

    def __init__(
        self, config_path: Path | str = CONFIG_PATH, *, attempt_id: str,
        mode: str, output_root: Path | None = None, ledger_path: Path = LEDGER,
        allow_real_api: bool = False, confirmation: str | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError("held-out mode must be fake, real, or replay")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id):
            raise ValueError("unsafe held-out attempt ID")
        self.config = HeldoutConfig.load(config_path)
        self.attempt_id, self.mode = attempt_id, mode
        self.allow_real_api, self.confirmation = allow_real_api, confirmation
        self.ledger_path = Path(ledger_path)
        self.provider_factory = provider_factory
        self.execution_status = "FAKE_PROVIDER_TEST" if mode == "fake" else "REAL_PROVIDER"
        self.validation_status = "OFFLINE_PROTOCOL_VALIDATION" if mode == "fake" else "REAL_EXPERIMENT"
        self.git_state = _git_state()
        self.source_snapshot = heldout_source_snapshot(self.config)
        self.registry = CanonicalToolRegistry.from_displayed_spec()
        self.builder = HeldoutEvidenceViewBuilder()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.output_schema = json.loads(OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.record_validator = Draft202012Validator(json.loads(RECORD_SCHEMA_PATH.read_text()))
        self.manifest_validator = Draft202012Validator(json.loads(MANIFEST_SCHEMA_PATH.read_text()))
        self.summary_validator = Draft202012Validator(json.loads(SUMMARY_SCHEMA_PATH.read_text()))
        self.network_calls = 0
        self.cache_misses = 0
        self.provider_instances = 0
        self.real_provider_instances = 0
        self._evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
        formal_root = FORMAL_ATTEMPT_ROOT / attempt_id
        self.root = Path(output_root or formal_root)
        if self.config.infrastructure_gate_version == 2 and attempt_id in self.config.raw["excluded_attempts"]:
            raise HeldoutProtocolError("V2 refuses the excluded incomplete V1 Attempt ID")
        if mode == "fake" and self._is_within(self.root, FORMAL_ATTEMPT_ROOT):
            raise PermissionError("FakeProvider validation output must not enter the formal real-attempt root")
        if mode == "real":
            self._validate_real_authorization()
        elif allow_real_api or confirmation is not None:
            raise PermissionError("offline held-out modes reject real-API authorization flags")
        if mode == "fake" and self.config.run_authorized:
            raise PermissionError("FakeProvider validation requires run_authorized:false")
        if mode == "replay" and not self.config.raw["execution"]["replay_enabled"]:
            raise PermissionError("held-out replay is disabled")
        self.results = self.root / ("replay" if mode == "replay" else "results")
        self._load_original_identity_for_replay()
        current, current_hash = _validated_ledger(self.ledger_path)
        self.invocation_ledger_before = _public_ledger_snapshot(current)
        self.invocation_ledger_before_hash = current_hash
        self.ledger_before, self.ledger_before_hash = self._attempt_ledger_baseline()
        settled = self._settled_attempt_cost()
        self.costs = (
            AtomicFocusedLedger(
                self.ledger_path, _HeldoutLedgerConfigAdapter(self.config), attempt_id,
                self.ledger_before, settled, pricing_catalog=PricingCatalog(
                    json.loads(PRICING_PATH.read_text(encoding="utf-8"))
                ),
            ) if mode == "real" else None
        )

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
        config = HeldoutConfig.load(config_path)
        return {
            "passed": True, "experiment_mode": EXPERIMENT_MODE,
            "run_authorized": config.run_authorized,
            "records": len(config.plan), "counts": config.counts,
            "development_family_count": len(set(item.family for item in config.plan) & set(DEVELOPMENT_FAMILIES)),
            "heldout_family_count": len({item.family for item in config.plan}),
            "heldout_overlap": 0,
            "repetitions": 1,
            "real_api_status": "NOT RUN",
        }

    def _validate_real_authorization(self) -> None:
        expected_confirmation = (
            CONFIRM_432_V2 if self.config.infrastructure_gate_version == 2 else CONFIRM_432
        )
        if not (
            self.config.run_authorized and self.allow_real_api
            and self.confirmation == expected_confirmation
        ):
            raise PermissionError(
                "real held-out requires run_authorized:true, --allow-real-api, and "
                + expected_confirmation
            )
        if self.provider_factory is not None:
            raise PermissionError("formal real held-out forbids injected or Fake providers")
        if self.git_state["dirty"]:
            raise PermissionError("formal held-out requires a clean Git worktree")
        missing = [model.provider for model in self.config.models if not model_api_key(model)]
        if missing:
            raise PermissionError("unconfigured held-out Provider credentials: " + ", ".join(missing))

    def _load_original_identity_for_replay(self) -> None:
        self.original_manifest: dict[str, Any] | None = None
        if self.mode != "replay":
            return
        path = self.root / "results/manifest.json"
        if not path.exists():
            raise FileNotFoundError("held-out replay requires an original Attempt Manifest")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.manifest_validator.validate(manifest)
        if manifest["attempt_id"] != self.attempt_id or manifest["config_hash"] != self.config.config_hash:
            raise HeldoutProtocolError("held-out replay identity does not match the original Attempt")
        if manifest["source_snapshot"] != self.source_snapshot:
            raise HeldoutProtocolError("held-out replay source snapshot changed")
        if manifest["plan"] != [item.to_dict() for item in self.config.plan]:
            raise HeldoutProtocolError("held-out replay plan changed")
        if manifest["counts"] != self.config.counts:
            raise HeldoutProtocolError("held-out replay balance counts changed")
        if manifest["tool_registry"] != self.registry.to_dict():
            raise HeldoutProtocolError("held-out replay Tool Registry changed")
        if manifest["analysis_plan_sha256"] != _sha256_bytes(ANALYSIS_PLAN_PATH):
            raise HeldoutProtocolError("held-out replay analysis plan changed")
        self.original_manifest = manifest
        self.execution_status = manifest["execution_status"]
        self.validation_status = manifest["validation_status"]

    def _attempt_ledger_baseline(self) -> tuple[dict[str, Any], str]:
        if self.original_manifest is None:
            return self.invocation_ledger_before, self.invocation_ledger_before_hash
        baseline = _public_ledger_snapshot(self.original_manifest["ledger_before"])
        if not _ledger_dominates(self.invocation_ledger_before, baseline):
            raise HeldoutProtocolError("formal Ledger rolled back below the held-out Attempt baseline")
        return baseline, self.original_manifest["ledger_before_sha256"]

    def _settled_attempt_cost(self) -> float:
        if self.mode != "real":
            return 0.0
        total = 0.0
        for path in (self.root / "results/checkpoint").glob("*/*.json"):
            total += float(json.loads(path.read_text()).get("cost_cny_delta", 0.0))
        return total

    def _prepare_evidence(self) -> None:
        if self._evidence:
            return
        for family in HELDOUT_FAMILIES:
            for variant in VARIANTS:
                paired: dict[str, dict[str, Any]] = {}
                for view in EVIDENCE_VIEWS:
                    evidence = self.builder.build(family, variant, view)
                    paired[view] = evidence
                    self._evidence[(family, variant, view)] = evidence
                validate_paired_evidence(paired)

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "schema_version": "specdriftbench-heldout432-checkpoint-identity-v1",
            "experiment_mode": EXPERIMENT_MODE, "attempt_id": self.attempt_id,
            "execution_status": self.execution_status, "config_hash": self.config.config_hash,
            "source_snapshot": self.source_snapshot, "plan_records": 432,
            "infrastructure_gate_version": self.config.infrastructure_gate_version,
            "protocol_amendment": self.config.raw.get("protocol_amendment"),
        }

    def _bind_checkpoint(self, resume: bool) -> None:
        path = self.results / "checkpoint/identity.json"
        expected = self._checkpoint_identity()
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise HeldoutProtocolError("held-out checkpoint identity changed")
            if not resume:
                raise FileExistsError("held-out checkpoint exists; pass --resume")
        else:
            if resume:
                raise FileNotFoundError("held-out resume requested without checkpoint identity")
            _atomic_json(path, expected)

    def _cache_namespace(self) -> str:
        if self.execution_status == "FAKE_PROVIDER_TEST":
            return self.config.raw["execution"]["fake_cache_namespace"]
        return self.config.raw["execution"]["cache_namespace"]

    def _cache_root(self) -> Path:
        return self.root / ("fake_cache" if self.execution_status == "FAKE_PROVIDER_TEST" else "cache")

    def _cache(self, provider: str) -> LLMCache:
        namespace = self._cache_namespace()
        directory = self._cache_root() / provider / namespace
        identity_path = directory / "identity.json"
        identity = {
            "schema_version": "specdriftbench-heldout432-cache-identity-v1",
            "experiment_mode": EXPERIMENT_MODE, "execution_status": self.execution_status,
            "attempt_id": self.attempt_id, "provider": provider,
            "config_hash": self.config.config_hash, "source_snapshot": self.source_snapshot,
            "cache_namespace": namespace,
            "canary_cache": False, "end_to_end_cache": False,
            "infrastructure_gate_version": self.config.infrastructure_gate_version,
            "excluded_v1_attempt": EXCLUDED_V1_ATTEMPT if self.config.infrastructure_gate_version == 2 else None,
        }
        if self.mode == "replay":
            if not identity_path.exists():
                raise HeldoutCacheMiss(f"held-out Cache identity missing for {provider}")
            if json.loads(identity_path.read_text()) != identity:
                raise HeldoutProtocolError("held-out replay Cache identity mismatch")
        elif identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise HeldoutProtocolError("held-out Cache identity mismatch")
        elif not identity_path.exists():
            _atomic_json(identity_path, identity)
        return LLMCache(directory)

    def _provider(self, model: ModelConfig) -> LLMProvider:
        self.provider_instances += 1
        if self.mode == "fake":
            return self.provider_factory(model) if self.provider_factory else HeldoutFakeProvider(model)
        key = model_api_key(model)
        if not key:
            raise PermissionError(f"{model.provider} credential is unconfigured")
        self.real_provider_instances += 1
        return OpenAICompatibleProvider(
            model, key, rate_limiter=RequestRateLimiter(
                float(self.config.raw["execution"]["rate_limit_per_second"])
            ), cost_controller=self.costs,
        )

    def _manifest(self) -> dict[str, Any]:
        if self.original_manifest is not None:
            return self.original_manifest
        manifest = {
            "schema_version": "specdriftbench-component-heldout-manifest-v1",
            "experiment_mode": EXPERIMENT_MODE,
            "execution_status": self.execution_status,
            "validation_status": self.validation_status,
            "attempt_id": self.attempt_id,
            "git_commit": self.git_state["commit"], "git_tree": self.git_state["tree"],
            "git_dirty": self.git_state["dirty"], "source_snapshot": self.source_snapshot,
            "config_hash": self.config.config_hash,
            "analysis_plan_sha256": _sha256_bytes(ANALYSIS_PLAN_PATH),
            "frozen_fingerprints": dict(FROZEN_FINGERPRINTS),
            "tool_registry": self.registry.to_dict(),
            "provider_capability_profiles": {
                model.provider: ProviderCapabilityAdapter.for_model(model).public_profile()
                for model in self.config.models
            },
            "plan": [item.to_dict() for item in self.config.plan],
            "counts": self.config.counts,
            "development_families": list(DEVELOPMENT_FAMILIES),
            "heldout_families": list(HELDOUT_FAMILIES), "heldout_overlap": 0,
            "repetitions": 1, "provider_order": list(PROVIDER_ORDER),
            "cache_namespaces": {
                provider: f"{provider}/{self._cache_namespace()}" for provider in PROVIDER_ORDER
            },
            "result_namespace": self.config.raw["execution"]["result_namespace"],
            "ledger_before": self.ledger_before,
            "ledger_before_sha256": self.ledger_before_hash,
        }
        if self.config.infrastructure_gate_version == 2:
            manifest.update({
                "infrastructure_gate_version": 2,
                "protocol_amendment": "1.1",
                "excluded_attempts": list(self.config.raw["excluded_attempts"]),
            })
        return manifest

    def _write_manifest(self) -> None:
        manifest = self._manifest()
        self.manifest_validator.validate(manifest)
        path = self.results / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise HeldoutProtocolError("held-out Manifest identity changed")
        _atomic_json(path, manifest)

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        self._prepare_evidence()
        self._bind_checkpoint(resume)
        self._write_manifest()
        completed: list[dict[str, Any]] = []
        gates: list[dict[str, Any]] = []
        self._write_summary(completed, gates, "INCOMPLETE")
        stop_attempt = False
        for provider_name in PROVIDER_ORDER:
            model = self.config.model(provider_name)
            cache = self._cache(provider_name)
            boundary = self._provider(model) if self.mode != "replay" else None
            provider_records: list[dict[str, Any]] = []
            gate_v2 = (
                InfrastructureGateV2(provider_name)
                if self.config.infrastructure_gate_version == 2 else None
            )
            for plan in (item for item in self.config.plan if item.provider == provider_name):
                checkpoint = self.results / "checkpoint" / provider_name / f"{plan.record_id}.json"
                if checkpoint.exists():
                    if not resume:
                        raise FileExistsError(f"held-out checkpoint exists for {plan.record_id}")
                    record = json.loads(checkpoint.read_text(encoding="utf-8"))
                    self._verify_record_identity(plan, record)
                else:
                    record = self._run_record(plan, model, cache, boundary)
                    _atomic_json(checkpoint, record)
                if self.mode == "replay":
                    self._verify_replay_record(plan, record)
                self.record_validator.validate(record)
                _atomic_json(self.results / "records" / f"{plan.record_id}.json", record)
                completed.append(record)
                provider_records.append(record)
                self._write_summary(completed, gates, "INCOMPLETE")
                if gate_v2 is not None:
                    gate = gate_v2.observe(record)
                    _atomic_json(self.results / "provider_gates" / f"{provider_name}.json", gate)
                    if gate["stopped"]:
                        stop_attempt = True
                        break
                elif (
                    self._is_infrastructure(record)
                    and sum(self._is_infrastructure(row) for row in provider_records)
                    >= int(self.config.raw["execution"]["infrastructure_error_limit"])
                ):
                    stop_attempt = True
                    break
            gate = gate_v2.public_dict() if gate_v2 is not None else {
                    "provider": provider_name, "completed": len(provider_records),
                    "infrastructure_errors": sum(self._is_infrastructure(row) for row in provider_records),
                    "expected": 144,
                }
            _atomic_json(self.results / "provider_gates" / f"{provider_name}.json", gate)
            gates.append(gate)
            if stop_attempt:
                break
        status = "COMPLETE" if len(completed) == 432 else "STOPPED"
        summary = self._write_summary(completed, gates, status)
        if self.mode in {"fake", "replay"}:
            _, digest = _validated_ledger(self.ledger_path)
            if digest != self.invocation_ledger_before_hash:
                raise RuntimeError("offline held-out execution changed the formal Ledger")
        if self.mode == "real" and float(summary["reserved_cny_after"]) != 0.0:
            raise RuntimeError("real held-out completed with an outstanding reservation")
        return {"summary": summary, "records": completed}

    def _run_record(
        self, plan: HeldoutPlan, model: ModelConfig, cache: LLMCache,
        provider: LLMProvider | None,
    ) -> dict[str, Any]:
        evidence = deepcopy(self._evidence[(plan.family, plan.variant, plan.evidence_view)])
        assert_heldout_provider_visible(evidence)
        output_schema_text = json.dumps(self.output_schema, indent=2, sort_keys=True)
        system = self.prompt.replace("{{OUTPUT_SCHEMA}}", output_schema_text)
        messages: tuple[dict[str, str], ...] = (
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, default=list)},
        )
        assert_heldout_provider_visible(messages)
        prompt_hash = hashlib.sha256(system.encode()).hexdigest()
        schema_hash = _sha256_bytes(OUTPUT_SCHEMA_PATH)
        responses: list[ProviderResponse] = []
        parsed = normalized = None
        error: BaseException | None = None
        raw_ref = raw_sha = None
        format_repairs = 0
        logical_boundary_calls = 0
        record_ledger_before = _public_ledger_snapshot(_validated_ledger(self.ledger_path, allow_reserved=True)[0])
        for repair_index in range(2):
            logical_boundary_calls += 1
            request = ProviderRequest(
                messages, self.output_schema, prompt_hash, evidence["public_scenario_id"],
                5, "component_attribution", plan.repetition,
                self.config.raw["execution"]["seed"],
                f"specdriftbench_heldout432_{plan.evidence_view.lower()}", self.config.config_hash,
            )
            key = LLMCache.key(model, request, schema_hash)
            response = cache.get(key)
            try:
                if response is None:
                    if self.mode == "replay":
                        self.cache_misses += 1
                        raise HeldoutCacheMiss(
                            f"held-out Cache miss for {plan.provider}/{plan.family}/{plan.variant}/{plan.evidence_view}"
                        )
                    if provider is None:
                        raise RuntimeError("held-out Provider boundary is unavailable")
                    response = provider.complete(request)
                    cache.put(key, response, model_api_key(model) if self.mode == "real" else None)
                responses.append(response)
                raw_ref = f"{plan.provider}/{self._cache_namespace()}/raw/{key}.json"
                raw_sha = hashlib.sha256(response.raw_text.encode()).hexdigest()
                assert_heldout_provider_visible(response.raw_text)
                if response.finish_reason == "length":
                    raise OutputTruncated(response.finish_reason, len(response.raw_text))
                parsed = parse_structured_output(response.raw_text, self.output_schema)
                assert_heldout_provider_visible(parsed)
                normalized = normalize_prediction(parsed, self.registry)
                if not response.cached:
                    response = replace(response, parsed_output=parsed)
                    cache.put(key, response, model_api_key(model) if self.mode == "real" else None)
                    responses[-1] = response
                break
            except InvalidStructuredOutput as exc:
                error = exc
                if repair_index == 1:
                    break
                format_repairs = 1
                messages = messages + (
                    {"role": "assistant", "content": response.raw_text if response else "{}"},
                    {"role": "user", "content": "Schema validation error: " + str(exc) + "\nReturn one corrected JSON object matching the unchanged schema."},
                )
                assert_heldout_provider_visible(messages)
            except Exception as exc:
                error = exc
                break
        if normalized is not None:
            error = None
        error_class = BenchmarkErrorClass.NONE if error is None else classify_error(error)
        if self.mode == "replay" and isinstance(error, HeldoutCacheMiss):
            raise error
        expected = expected_for_evaluator(self.builder, plan.family, plan.variant)
        evaluation = self._evaluate(normalized, expected)
        after = _public_ledger_snapshot(_validated_ledger(self.ledger_path, allow_reserved=True)[0])
        real_attempts = int(after["api_attempts"] - record_ledger_before["api_attempts"])
        ledger_input_tokens = int(
            after["provider_reported_input_tokens"]
            - record_ledger_before["provider_reported_input_tokens"]
        )
        ledger_output_tokens = int(
            after["provider_reported_output_tokens"]
            - record_ledger_before["provider_reported_output_tokens"]
        )
        response_attempts = sum(response.provider_attempts for response in responses)
        error_public = self._public_error(error)
        offline_error_attempts = (
            int((error_public or {}).get("actual_network_attempts", 0))
            if isinstance(error, ProviderError) else 0
        )
        actual_network_attempts = (
            real_attempts if self.mode == "real"
            else response_attempts + offline_error_attempts
        )
        input_tokens = ledger_input_tokens if self.mode == "real" else sum(
            response.input_tokens for response in responses
        )
        output_tokens = ledger_output_tokens if self.mode == "real" else sum(
            response.output_tokens for response in responses
        )
        reasoning_tokens = sum(response.reasoning_tokens for response in responses)
        latency_ms = sum(response.latency_ms for response in responses) + float(
            (error_public or {}).get("latency_ms", 0.0)
        )
        usage_if_available = (
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
            if responses or input_tokens or output_tokens else None
        )
        if self.mode == "real":
            self.network_calls += real_attempts
        record = {
            "schema_version": "specdriftbench-component-heldout-record-v1",
            "record_id": plan.record_id, "attempt_id": self.attempt_id,
            "experiment_mode": EXPERIMENT_MODE, "execution_status": self.execution_status,
            "provider": plan.provider, "model": plan.model,
            "provider_profile_id": ProviderCapabilityAdapter.for_model(model).profile_id,
            "max_tokens": model.max_output_tokens, "family": plan.family,
            "variant": plan.variant, "evidence_view": plan.evidence_view,
            "repetition": plan.repetition, "paired_group_id": plan.paired_group_id,
            "public_scenario_id": evidence["public_scenario_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "raw_response_ref": raw_ref, "raw_response_sha256": raw_sha,
            "parsed_result": parsed, "normalized_prediction": normalized,
            "schema_valid": normalized is not None, "format_repairs": format_repairs,
            "error_class": error_class.value, "error": error_public,
            "evaluation": evaluation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
            "provider_attempts": actual_network_attempts,
            "logical_boundary_calls": logical_boundary_calls,
            "actual_network_attempts": actual_network_attempts,
            "provider_retry_count": max(0, actual_network_attempts - logical_boundary_calls),
            "format_repair_count": format_repairs,
            "usage_if_available": usage_if_available,
            "finish_reason": responses[-1].finish_reason if responses else None,
            "content_length": len(responses[-1].raw_text) if responses else 0,
            "cost_cny_delta": max(0.0, float(after["spent_cny"]) - float(record_ledger_before["spent_cny"])) if self.mode == "real" else 0.0,
            "cache_hit": bool(responses) and all(response.cached for response in responses),
            "fallback_used": False, "main_state_pollution": False,
            "leakage": {"passed": error_class != BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE, "api_key_value_present": False, "hidden_fields": []},
            "termination_reason": "COMPONENT_COMPLETE" if error is None else error_class.value,
        }
        self.record_validator.validate(record)
        if self.mode == "real":
            key = model_api_key(model)
            if key:
                assert_secret_absent(record, key)
        return record

    @staticmethod
    def _evaluate(prediction: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> dict[str, bool | None]:
        if prediction is None:
            return {"class_correct": None, "category_correct": None, "target_correct": None, "location_correct": None}
        location = prediction.get("normalized_location") or {}
        expected_location = expected.get("location")
        location_correct = True if expected_location is None else expected_location in {
            location.get("spec_pointer"), location.get("runtime_path"),
        }
        return {
            "class_correct": prediction.get("predicted_class") == expected["predicted_class"],
            "category_correct": prediction.get("drift_category") == expected["drift_category"],
            "target_correct": prediction.get("target_tool") == expected["target_tool"],
            "location_correct": location_correct,
        }

    @staticmethod
    def _public_error(error: BaseException | None) -> dict[str, Any] | None:
        if error is None:
            return None
        if isinstance(error, ProviderError):
            return error.public_dict()
        return {"type": type(error).__name__, "sanitized_message": str(error)[:512]}

    @staticmethod
    def _is_infrastructure(record: Mapping[str, Any]) -> bool:
        return record["error_class"] in {
            BenchmarkErrorClass.EVALUATOR_ERROR.value,
            BenchmarkErrorClass.CONTROLLER_ERROR.value,
            BenchmarkErrorClass.NETWORK_ERROR.value,
            BenchmarkErrorClass.PROVIDER_ERROR.value,
        }

    @staticmethod
    def _verify_record_identity(plan: HeldoutPlan, record: Mapping[str, Any]) -> None:
        expected = plan.to_dict()
        for key in ("record_id", "paired_group_id", "provider", "model", "family", "variant", "evidence_view", "repetition"):
            if record.get(key) != expected[key]:
                raise HeldoutProtocolError(f"resumed held-out record identity changed: {key}")

    def _verify_replay_record(self, plan: HeldoutPlan, replay: Mapping[str, Any]) -> None:
        original_path = self.root / "results/records" / f"{plan.record_id}.json"
        if not original_path.exists():
            raise HeldoutCacheMiss(f"original held-out record missing for {plan.record_id}")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if original.get("fallback_used") is not False:
            raise HeldoutProtocolError("held-out replay refuses an original fallback record")
        for field in _SEMANTIC_REPLAY_FIELDS:
            if replay.get(field) != original.get(field):
                raise HeldoutProtocolError(f"held-out replay semantic mismatch for {plan.record_id}: {field}")

    def _write_summary(
        self, records: list[dict[str, Any]], gates: list[dict[str, Any]], status: str,
    ) -> dict[str, Any]:
        current, digest = _validated_ledger(self.ledger_path, allow_reserved=self.mode == "real")
        current_public = _public_ledger_snapshot(current)
        delta = self.ledger_before if self.mode == "real" else self.invocation_ledger_before
        summary = {
            "schema_version": "specdriftbench-component-heldout-summary-v1",
            "experiment_mode": EXPERIMENT_MODE, "execution_status": self.execution_status,
            "attempt_id": self.attempt_id, "mode": self.mode, "status": status,
            "records_planned": 432, "records_completed": len(records),
            "provider_instances": self.provider_instances,
            "real_provider_instances": self.real_provider_instances,
            "network_calls": self.network_calls, "cache_misses": self.cache_misses,
            "fallback_count": sum(bool(record["fallback_used"]) for record in records),
            "format_repair_count": sum(int(record["format_repairs"]) for record in records),
            "ledger_before": self.ledger_before, "ledger_after": current_public,
            "ledger_before_sha256": self.ledger_before_hash, "ledger_after_sha256": digest,
            "attempts_delta": int(current_public["api_attempts"] - delta["api_attempts"]),
            "input_tokens_delta": int(current_public["provider_reported_input_tokens"] - delta["provider_reported_input_tokens"]),
            "output_tokens_delta": int(current_public["provider_reported_output_tokens"] - delta["provider_reported_output_tokens"]),
            "cost_cny_delta": float(current_public["spent_cny"] - delta["spent_cny"]),
            "reserved_cny_after": current_public["reserved_cny"],
            "provider_gates": gates,
        }
        self.summary_validator.validate(summary)
        _atomic_json(self.results / "summary.json", summary)
        return summary


def run_offline_fake_validation(
    output_root: Path, *, attempt_id: str = "heldout432-fake-validation-v1",
    config_path: Path | str = CONFIG_PATH, ledger_path: Path = LEDGER,
) -> dict[str, Any]:
    runner = HeldoutRunner(
        config_path, attempt_id=attempt_id, mode="fake",
        output_root=output_root, ledger_path=ledger_path,
    )
    result = runner.run()
    return {
        "protocol_status": "FAKE_PROVIDER_TEST",
        "validation_status": "OFFLINE_PROTOCOL_VALIDATION",
        **result,
    }
