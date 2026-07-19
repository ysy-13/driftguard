from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


IMMEDIATE_STATUSES = frozenset({401, 402})
RECORD_REQUEST_STATUSES = frozenset({400, 403, 404, 405, 409, 413, 415, 422})
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
IMMEDIATE_MARKERS = (
    "invalid api key", "invalid_api_key", "authentication failed",
    "permission denied", "access denied",
    "insufficient balance", "insufficient_balance", "insufficient quota",
    "model not found", "model_not_found", "invalid model",
    "unsupported protocol", "invalid endpoint", "endpoint configuration",
)


def _error_text(error: Mapping[str, Any]) -> str:
    return " ".join(
        str(error.get(key) or "").lower()
        for key in ("provider_error_code", "sanitized_message", "exception_type")
    )


def classify_gate_record(record: Mapping[str, Any]) -> str:
    """Classify one final record for the frozen V2 infrastructure gate."""
    leakage = record.get("leakage")
    if (
        (isinstance(leakage, Mapping) and leakage.get("passed") is False)
        or bool(record.get("fallback_used"))
        or bool(record.get("main_state_pollution"))
    ):
        return "IMMEDIATE_PROTOCOL_SAFETY_ERROR"
    error = record.get("error")
    if not isinstance(error, Mapping):
        return "SUCCESS"
    status = error.get("status_code")
    status = int(status) if isinstance(status, int) else None
    text = _error_text(error)
    if status in IMMEDIATE_STATUSES or any(marker in text for marker in IMMEDIATE_MARKERS):
        return "IMMEDIATE_PERMANENT_PROVIDER_ERROR"
    if status in RECORD_REQUEST_STATUSES:
        return "RECORD_REQUEST_ERROR"
    error_class = str(record.get("error_class") or "")
    layer = str(error.get("failure_layer") or "")
    if status in TRANSIENT_STATUSES:
        return "FINAL_INFRASTRUCTURE_ERROR"
    if error_class == "NETWORK_ERROR" or layer in {"DNS", "CONNECTION", "TLS", "TIMEOUT"}:
        return "FINAL_INFRASTRUCTURE_ERROR"
    if error_class == "PROVIDER_ERROR" and status is None and layer != "RESPONSE_PARSE":
        return "FINAL_INFRASTRUCTURE_ERROR"
    return "NON_INFRASTRUCTURE_ERROR"


@dataclass
class InfrastructureGateV2:
    provider: str
    completed: int = 0
    final_infrastructure_errors: int = 0
    consecutive_final_infrastructure_errors: int = 0
    stop: bool = False
    stop_reason: str | None = None

    @classmethod
    def from_records(
        cls, provider: str, records: list[Mapping[str, Any]],
    ) -> "InfrastructureGateV2":
        gate = cls(provider)
        for record in records:
            gate.observe(record)
        return gate

    def observe(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if self.stop:
            raise RuntimeError("cannot append records after the V2 Provider gate stopped")
        disposition = classify_gate_record(record)
        self.completed += 1
        if disposition == "FINAL_INFRASTRUCTURE_ERROR":
            self.final_infrastructure_errors += 1
            self.consecutive_final_infrastructure_errors += 1
        else:
            self.consecutive_final_infrastructure_errors = 0
        if disposition in {
            "IMMEDIATE_PERMANENT_PROVIDER_ERROR", "IMMEDIATE_PROTOCOL_SAFETY_ERROR",
        }:
            self.stop, self.stop_reason = True, disposition
        elif self.consecutive_final_infrastructure_errors >= 3:
            self.stop, self.stop_reason = True, "THREE_CONSECUTIVE_FINAL_INFRASTRUCTURE_ERRORS"
        elif self.completed >= 24 and self.infrastructure_error_rate >= 0.10:
            self.stop, self.stop_reason = True, "FINAL_INFRASTRUCTURE_ERROR_RATE_AT_LEAST_10_PERCENT"
        return {**self.public_dict(), "last_record_disposition": disposition}

    @property
    def infrastructure_error_rate(self) -> float:
        return self.final_infrastructure_errors / self.completed if self.completed else 0.0

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "specdriftbench-heldout-infrastructure-gate-v2",
            "infrastructure_gate_version": 2,
            "protocol_amendment": "1.1",
            "provider": self.provider,
            "completed": self.completed,
            "expected": 144,
            "infrastructure_errors": self.final_infrastructure_errors,
            "final_infrastructure_errors": self.final_infrastructure_errors,
            "consecutive_final_infrastructure_errors": self.consecutive_final_infrastructure_errors,
            "infrastructure_error_rate": self.infrastructure_error_rate,
            "minimum_records_for_rate_gate": 24,
            "rate_gate_threshold": 0.10,
            "consecutive_gate_threshold": 3,
            "stopped": self.stop,
            "stop_reason": self.stop_reason,
        }


def descriptive_availability_metrics(records: list[Mapping[str, Any]]) -> dict[str, float | int]:
    """Frozen denominator handling for incomplete/non-evaluable held-out records."""
    total = len(records)
    evaluable = [row for row in records if row.get("normalized_prediction") is not None]
    responses = [row for row in records if row.get("raw_response_ref") is not None]
    class_correct = sum(row.get("evaluation", {}).get("class_correct") is True for row in records)
    schema_valid = sum(row.get("schema_valid") is True for row in responses)
    infrastructure_errors = sum(
        classify_gate_record(row) == "FINAL_INFRASTRUCTURE_ERROR" for row in records
    )
    return {
        "records": total,
        "response_available": len(responses),
        "evaluable": len(evaluable),
        "all_record_accuracy": class_correct / total if total else 0.0,
        "evaluable_accuracy": class_correct / len(evaluable) if evaluable else 0.0,
        "coverage": len(evaluable) / total if total else 0.0,
        "schema_validity_among_responses": schema_valid / len(responses) if responses else 0.0,
        "infrastructure_errors": infrastructure_errors,
        "infrastructure_error_rate": infrastructure_errors / total if total else 0.0,
    }
