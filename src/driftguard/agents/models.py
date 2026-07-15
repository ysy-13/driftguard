from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    FINAL_ANSWER = "FINAL_ANSWER"
    REQUEST_PROBE = "REQUEST_PROBE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    concise_decision_summary: str
    tool_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    probe_type: str | None = None
    target_tool_id: str | None = None
    hypothesis: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentAction":
        action_type = ActionType(value["action_type"])
        if action_type == ActionType.TOOL_CALL and (not value.get("tool_id") or not isinstance(value.get("arguments"), dict)):
            raise ValueError("TOOL_CALL requires tool_id and arguments")
        if action_type == ActionType.FINAL_ANSWER and not isinstance(value.get("answer"), str):
            raise ValueError("FINAL_ANSWER requires answer")
        if action_type == ActionType.REQUEST_PROBE and not all(value.get(key) for key in ("probe_type", "target_tool_id", "hypothesis")):
            raise ValueError("REQUEST_PROBE is incomplete")
        return cls(
            action_type, str(value.get("concise_decision_summary", "")), value.get("tool_id"),
            dict(value.get("arguments", {})), value.get("answer"), value.get("probe_type"),
            value.get("target_tool_id"), value.get("hypothesis"), tuple(value.get("evidence_refs", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action_type"] = self.action_type.value
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class LLMAttribution:
    predicted_class: str
    target_tool_id: str | None
    drift_category: str
    location_type: str
    location_path: str | None
    evidence_refs: tuple[str, ...]
    requested_probe: dict[str, Any] | None
    confidence: float
    concise_reason: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LLMAttribution":
        predicted = value["predicted_class"]
        category = value["drift_category"]
        if predicted not in {"AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT", "INSUFFICIENT_EVIDENCE"}:
            raise ValueError("unsupported attribution class")
        if category not in {"ICD", "RSD", "WPD", "SED", "UNKNOWN"}:
            raise ValueError("unsupported drift category")
        return cls(
            predicted, value.get("target_tool_id"), category, str(value["location_type"]),
            value.get("location_path"), tuple(value.get("evidence_refs", ())),
            value.get("requested_probe"), float(value["confidence"]), str(value["concise_reason"]),
        )

    def validate_evidence(self, available_refs: set[str]) -> None:
        if not set(self.evidence_refs).issubset(available_refs):
            raise ValueError("LLM attribution cites unknown evidence")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class LLMPatchProposal:
    proposal: Any
    raw_attribution: LLMAttribution
    guard_requested: bool = True

    def __post_init__(self) -> None:
        if not self.guard_requested:
            raise ValueError("LLM patches must request deterministic validation")


@dataclass(frozen=True)
class AgentRunResult:
    task_success: bool
    termination_reason: str
    error_category: str | None
    answer: str | None
    tool_calls: int
    probe_calls: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    trace_references: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    task_evaluation: dict[str, Any]
    state_transitions: tuple[dict[str, Any], ...] = ()
    catalog_fingerprint: str = ""
    policy_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["trace_references"] = list(self.trace_references)
        value["actions"] = list(self.actions)
        value["state_transitions"] = list(self.state_transitions)
        value["policy_events"] = list(self.policy_events)
        return value
