from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class PatchLifecycle(str, Enum):
    PROPOSED = "PROPOSED"
    STATIC_VALIDATED = "STATIC_VALIDATED"
    REPAIR_VALIDATED = "REPAIR_VALIDATED"
    REGRESSION_VALIDATED = "REGRESSION_VALIDATED"
    SAFETY_VALIDATED = "SAFETY_VALIDATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: str
    value: Any
    evidence_refs: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        if self.op not in {"add", "remove", "replace", "move", "copy", "test"}:
            raise ValueError(f"unsupported RFC 6902 operation: {self.op}")
        if not self.path.startswith("/"):
            raise ValueError("patch operation path must be a JSON Pointer")
        if not self.evidence_refs:
            raise ValueError("every patch operation must cite visible evidence")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    stage: str
    reason_codes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "stage": self.stage,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }
        return payload


@dataclass(frozen=True)
class ToolSpecPatch:
    patch_id: str
    patch_version: str
    created_at: str
    target_tool_id: str
    source_spec_fingerprint: str
    drift_category: str
    location_type: str
    location_path: str
    evidence_refs: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    openapi_operations: tuple[PatchOperation, ...]
    semantic_extensions: dict[str, Any]
    expected_agent_behavior_change: str
    confidence: float
    lifecycle_status: PatchLifecycle = PatchLifecycle.PROPOSED
    validation_results: tuple[ValidationResult, ...] = ()
    normalized_location: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_refs or not self.openapi_operations:
            raise ValueError("patches require evidence and at least one operation")
        if self.drift_category not in {"ICD", "RSD", "WPD", "SED"}:
            raise ValueError("unsupported drift category")
        operation_refs = {ref for operation in self.openapi_operations for ref in operation.evidence_refs}
        if not operation_refs.issubset(set(self.evidence_refs)):
            raise ValueError("operation evidence must be included in patch evidence")

    def with_status(self, status: PatchLifecycle, result: ValidationResult | None = None) -> "ToolSpecPatch":
        results = self.validation_results + ((result,) if result is not None else ())
        return replace(self, lifecycle_status=status, validation_results=results)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "patch_id": self.patch_id,
            "patch_version": self.patch_version,
            "created_at": self.created_at,
            "target_tool_id": self.target_tool_id,
            "source_spec_fingerprint": self.source_spec_fingerprint,
            "drift_category": self.drift_category,
            "location_type": self.location_type,
            "location_path": self.location_path,
            "evidence_refs": list(self.evidence_refs),
            "rationale_codes": list(self.rationale_codes),
            "openapi_operations": [operation.to_dict() for operation in self.openapi_operations],
            "semantic_extensions": self.semantic_extensions,
            "expected_agent_behavior_change": self.expected_agent_behavior_change,
            "confidence": self.confidence,
            "lifecycle_status": self.lifecycle_status.value,
            "validation_results": [result.to_dict() for result in self.validation_results],
        }
        if self.normalized_location:
            payload["normalized_location"] = dict(self.normalized_location)
        return payload


@dataclass(frozen=True)
class GenerationResult:
    disposition: str
    candidates: tuple[ToolSpecPatch, ...] = ()
    reason_codes: tuple[str, ...] = ()
