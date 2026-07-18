from __future__ import annotations

from copy import deepcopy
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping

from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.contracts.loader import load_openapi
from driftguard.evidence.leakage_guard import EvidenceLeakageError
from driftguard.evidence.models import thaw
from driftguard.experiments.budgets import BudgetExhausted
from driftguard.llm import InvalidStructuredOutput, ProviderError
from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner


FAMILIES = ("M01", "M06", "M11", "M16")
VARIANTS = ("AE", "TF", "PD")
EVIDENCE_VIEWS = ("FIRST_FAILURE", "RETRY_HISTORY", "FULL_EVIDENCE")
LEGAL_LABELS = {
    "AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT",
    "ICD", "RSD", "WPD", "SED", "NONE",
}
HIDDEN_KEYS = {
    "evaluator_metadata", "ground_truth_label", "variant", "variant_code",
    "source_drift_id", "expected_patch", "expected_patch_ref",
    "runtime_contract", "runtime_profile", "verification_token",
    "future_episode_evidence", "expected_action", "persistent_patch_allowed",
}
HIDDEN_TEXT = (
    "evaluator_metadata", "ground_truth_label", "source_drift_id",
    "expected_patch", "hidden runtime contract", "verification_token",
    "future_episode_evidence",
)


class BenchmarkErrorClass(str, Enum):
    NONE = "NONE"
    MODEL_ERROR = "MODEL_ERROR"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    CONTROLLER_ERROR = "CONTROLLER_ERROR"
    METHOD_BUDGET_FAILURE = "METHOD_BUDGET_FAILURE"
    NETWORK_ERROR = "NETWORK_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SAFETY_REJECTION = "SAFETY_REJECTION"
    TRUE_EVIDENCE_LEAKAGE = "TRUE_EVIDENCE_LEAKAGE"


def classify_error(exc: BaseException) -> BenchmarkErrorClass:
    if isinstance(exc, TrueEvidenceLeakage):
        return BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE
    if isinstance(exc, BudgetExhausted):
        return BenchmarkErrorClass.METHOD_BUDGET_FAILURE
    if isinstance(exc, OutputTruncated):
        return BenchmarkErrorClass.OUTPUT_TRUNCATED
    if isinstance(exc, InvalidStructuredOutput):
        return BenchmarkErrorClass.INVALID_STRUCTURED_OUTPUT
    if isinstance(exc, ProviderError):
        return (
            BenchmarkErrorClass.NETWORK_ERROR
            if exc.failure_layer.value in {"DNS", "CONNECTION", "TLS", "TIMEOUT"}
            else BenchmarkErrorClass.PROVIDER_ERROR
        )
    if isinstance(exc, EvidenceLeakageError):
        return BenchmarkErrorClass.EVALUATOR_ERROR
    return BenchmarkErrorClass.CONTROLLER_ERROR


class TrueEvidenceLeakage(EvidenceLeakageError):
    pass


class OutputTruncated(RuntimeError):
    def __init__(self, finish_reason: str, content_length: int):
        self.finish_reason = finish_reason
        self.content_length = content_length
        super().__init__(
            f"provider output truncated: finish_reason={finish_reason}, content_length={content_length}"
        )


def assert_benchmark_visible(value: Any, path: str = "$") -> None:
    """Versioned benchmark guard: legal prediction labels are not leakage."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in HIDDEN_KEYS:
                raise TrueEvidenceLeakage(f"hidden benchmark field at {path}.{key}")
            assert_benchmark_visible(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_benchmark_visible(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if value.upper() in LEGAL_LABELS:
            return
        if any(term in lowered for term in HIDDEN_TEXT):
            raise TrueEvidenceLeakage(f"hidden benchmark content at {path}")
        if re.search(r"\b(?:M\d{2}|(?:ICD|RSD|WPD|SED)-\d{2})\b", value, re.IGNORECASE):
            raise TrueEvidenceLeakage(f"family/source lookup content at {path}")


def balanced_evidence_view(family: str, variant: str) -> str:
    family_index, variant_index = FAMILIES.index(family), VARIANTS.index(variant)
    return EVIDENCE_VIEWS[(family_index + variant_index) % 3]


class CanonicalToolRegistry:
    def __init__(self, aliases: Mapping[str, str]):
        self.alias_to_tool = dict(aliases)
        self.tool_to_alias = {tool: alias for alias, tool in self.alias_to_tool.items()}
        if len(self.alias_to_tool) != len(self.tool_to_alias):
            raise ValueError("Tool Registry mapping is not bijective")

    @classmethod
    def from_displayed_spec(cls, spec: Mapping[str, Any] | None = None) -> "CanonicalToolRegistry":
        catalog = ToolCatalogRenderer().render(deepcopy(dict(spec or load_openapi())))
        return cls({f"T{index:02d}": item["operation_id"] for index, item in enumerate(catalog["tools"], 1)})

    def canonicalize(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value)
        alias = text.upper()
        if alias in self.alias_to_tool:
            return self.alias_to_tool[alias]
        return text if text in self.tool_to_alias else text

    def alias(self, tool_id: str) -> str:
        return self.tool_to_alias[tool_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "specdriftbench-tool-registry-v1",
            "aliases": dict(sorted(self.alias_to_tool.items())),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def normalize_prediction(value: Mapping[str, Any], registry: CanonicalToolRegistry) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["target_tool"] = registry.canonicalize(result.get("target_tool"))
    location = result.get("normalized_location")
    if isinstance(location, Mapping):
        location = deepcopy(dict(location))
        location["tool_id"] = registry.canonicalize(location.get("tool_id"))
        result["normalized_location"] = location
    return result


class EvidenceViewBuilder:
    FIRST_TYPES = {"task_received", "tool_call_proposed", "local_validation", "tool_response"}
    RETRY_TYPES = FIRST_TYPES | {"history_retrieval", "retry_result"}

    def __init__(self) -> None:
        self.runner = AttributionConformanceRunner(strict_leakage_check=False)
        self.families = {
            family["matched_case_id"]: family for family in self.runner.matched["families"]
            if family["matched_case_id"] in FAMILIES
        }

    def build(self, family_id: str, variant: str, view_name: str | None = None) -> dict[str, Any]:
        family = self.families[family_id]
        scenario = next(item for item in family["scenarios"] if item["variant_code"] == variant)
        artifacts = self.runner._run_scenario(family, scenario, return_agent_artifacts=True)
        trace = artifacts["agent_view"].trace.to_dict()
        selected = view_name or balanced_evidence_view(family_id, variant)
        events = [event for event in trace["events"] if self._include(event, selected)]
        evidence = {
            "schema_version": "specdriftbench-agent-view-v1",
            "evidence_view": selected,
            "public_scenario_id": artifacts["public_scenario_id"],
            "user_intent": scenario["agent_visible"]["evidence_bundle"]["user_task"],
            "displayed_specification": thaw(artifacts["agent_view"].displayed_spec),
            "evidence_trace": {**{key: deepcopy(child) for key, child in trace.items() if key != "events"}, "events": events},
        }
        assert_benchmark_visible(evidence)
        evidence["evidence_sha256"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=list).encode()
        ).hexdigest()
        return evidence

    def plan(self) -> list[dict[str, Any]]:
        return [
            {"family": family, "variant": variant, "evidence_view": balanced_evidence_view(family, variant)}
            for family in FAMILIES for variant in VARIANTS
        ]

    @classmethod
    def _include(cls, event: Mapping[str, Any], view_name: str) -> bool:
        event_type = event.get("event_type")
        if event_type == "diagnosis_emitted":
            return False
        if view_name == "FIRST_FAILURE":
            return event_type in cls.FIRST_TYPES and int(event.get("episode", 0)) <= 3
        if view_name == "RETRY_HISTORY":
            return event_type in cls.RETRY_TYPES and int(event.get("episode", 0)) <= 4
        return event_type in cls.RETRY_TYPES | {"probe_started", "probe_result"}


def expected_for_evaluator(builder: EvidenceViewBuilder, family_id: str, variant: str) -> dict[str, Any]:
    """Evaluator-only metadata; callers must never place this in Provider inputs or records."""
    family = builder.families[family_id]
    scenario = next(item for item in family["scenarios"] if item["variant_code"] == variant)
    case = builder.runner._case_by_id[family["source_drift_id"]]
    return {
        "predicted_class": {
            "agent_error": "AGENT_ERROR", "transient_failure": "TRANSIENT_FAILURE",
            "persistent_drift": "PERSISTENT_DRIFT",
        }[scenario["evaluator_metadata"]["ground_truth_label"]],
        "drift_category": {
            "input_contract": "ICD", "response_shape": "RSD",
            "workflow_precondition": "WPD", "state_effect": "SED",
        }[case["drift_type"]] if variant == "PD" else "NONE",
        "target_tool": case["target_tool"] if variant == "PD" else None,
        "location": case["ground_truth"]["location"] if variant == "PD" else None,
    }
