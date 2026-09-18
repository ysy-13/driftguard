from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CLASSES = ("AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT")


@dataclass(frozen=True)
class RulePrediction:
    predicted_class: str
    reason_code: str
    evidence_event_ids: tuple[str, ...]
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "reason_code": self.reason_code,
            "evidence_event_ids": list(self.evidence_event_ids),
            "used_fallback": self.used_fallback,
        }


class ProtocolRuleBaseline:
    """A transparent classifier over the same Agent-visible evidence as the LLMs.

    The classifier deliberately has no family, variant, provider, model, or
    evaluator input.  It operationalizes three protocol-level observations:
    direct visible contract nonconformance indicates Agent Error; recovery on
    an exact legal retry indicates Transient Failure; and persistence on that
    retry indicates Persistent Drift.  When the first-failure view contains no
    discriminating retry, it uses a declared conservative PD fallback so the
    baseline has the same mandatory three-class output coverage as the LLMs.
    """

    def predict(self, evidence: Mapping[str, Any]) -> RulePrediction:
        trace = evidence.get("evidence_trace")
        if not isinstance(trace, Mapping):
            raise ValueError("evidence_trace must be an object")
        events = trace.get("events")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise ValueError("evidence_trace.events must be an array")
        if not events:
            raise ValueError("evidence_trace.events must not be empty")
        if any(not isinstance(event, Mapping) for event in events):
            raise ValueError("every evidence event must be an object")

        agent_error_ids: list[str] = []
        for event in events:
            validation = event.get("local_validation_result")
            metadata = event.get("probe_metadata")
            if (
                event.get("event_type") == "local_validation"
                and isinstance(validation, Mapping)
                and validation.get("conforms_to_displayed_spec") is False
            ):
                agent_error_ids.append(self._event_id(event))
            if (
                isinstance(metadata, Mapping)
                and metadata.get("interpretation_status") == "failed"
            ):
                agent_error_ids.append(self._event_id(event))
        if agent_error_ids:
            return RulePrediction(
                "AGENT_ERROR",
                "VISIBLE_CALL_OR_INTERPRETATION_VIOLATES_DISPLAYED_CONTRACT",
                tuple(dict.fromkeys(agent_error_ids)),
            )

        retries = []
        for event in events:
            metadata = event.get("probe_metadata")
            if (
                event.get("event_type") == "retry_result"
                and isinstance(metadata, Mapping)
                and metadata.get("exact_retry") is True
            ):
                retries.append(event)
        if retries:
            retry = retries[-1]
            metadata = retry.get("probe_metadata")
            assert isinstance(metadata, Mapping)
            if metadata.get("success") is True:
                return RulePrediction(
                    "TRANSIENT_FAILURE",
                    "EXACT_LEGAL_RETRY_RECOVERED",
                    (self._event_id(retry),),
                )
            if metadata.get("success") is False:
                return RulePrediction(
                    "PERSISTENT_DRIFT",
                    "EXACT_LEGAL_RETRY_REPRODUCED_FAILURE",
                    (self._event_id(retry),),
                )
            raise ValueError("exact retry is missing a Boolean success outcome")

        failure_ids = tuple(
            self._event_id(event)
            for event in events
            if event.get("event_type") == "tool_response"
        )
        if not failure_ids:
            raise ValueError("evidence has neither a direct error nor a tool response")
        return RulePrediction(
            "PERSISTENT_DRIFT",
            "FIRST_FAILURE_CONSERVATIVE_PD_FALLBACK",
            failure_ids,
            used_fallback=True,
        )

    @staticmethod
    def _event_id(event: Mapping[str, Any]) -> str:
        value = event.get("event_id")
        if not isinstance(value, str) or not value:
            raise ValueError("evidence event is missing event_id")
        return value


def classification_metrics(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for actual, predicted in pairs:
        if actual not in CLASSES or predicted not in CLASSES:
            raise ValueError("rule-baseline metrics require legal three-class labels")
        matrix[actual][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in CLASSES if actual != label)
        fn = sum(matrix[label][predicted] for predicted in CLASSES if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    records = len(pairs)
    correct = sum(matrix[label][label] for label in CLASSES)
    return {
        "records": records,
        "correct": correct,
        "coverage": 1.0 if records else 0.0,
        "accuracy": correct / records if records else 0.0,
        "macro_precision": sum(float(row["precision"]) for row in per_class.values()) / 3,
        "macro_recall": sum(float(row["recall"]) for row in per_class.values()) / 3,
        "macro_f1": sum(float(row["f1"]) for row in per_class.values()) / 3,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
