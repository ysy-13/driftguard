from __future__ import annotations

from driftguard.contracts.loader import load_openapi
from driftguard.evidence import AgentView, EvidenceCollector, EvidenceStore


def build_view(kind: str):
    spec = load_openapi()
    store = EvidenceStore("trace-test", "public-safe-001")
    collector = EvidenceCollector(store, spec)
    collector.add("task_received", 1)
    collector.add(
        "local_validation", 3, "create_issue",
        displayed_request={"repo_id": "R1", "title": "x"},
        local_validation_result={"valid": kind != "AE", "scope": "request"},
    )
    collector.add(
        "tool_response", 3, "create_issue",
        displayed_request={"repo_id": "R1", "title": "x"},
        visible_runtime_response={"status_code": 422, "payload": {"ok": False}},
        normalized_observation={
            "channel": "response_error", "http_status": 422,
            "error_code": "VALIDATION_ERROR", "field": "priority",
            "message_pattern": "Missing required field: priority", "response_success": False,
        },
        probe_metadata={"independent_failure": kind in {"TF", "PD", "SINGLE"}},
    )
    if kind == "AE":
        collector.add("probe_result", 4, "create_issue", probe_metadata={
            "corrected_behavior": True, "success": True, "passed": True,
        })
    elif kind == "TF":
        collector.add("retry_result", 3, "create_issue", probe_metadata={
            "exact_retry": True, "success": True, "passed": True,
        })
    elif kind == "PD":
        collector.add("retry_result", 3, "create_issue", probe_metadata={
            "exact_retry": True, "success": False, "passed": True,
        })
        collector.add("probe_result", 4, "create_issue", probe_metadata={
            "independent_failure": True, "passed": True,
        })
        collector.add("probe_result", 5, "create_issue", probe_metadata={
            "discriminative": True, "passed": True,
        })
        for index in range(3):
            collector.add("probe_result", 5, "get_repository", probe_metadata={
                "regression_check": True, "success": True, "regression_index": index + 1,
            })
    return AgentView(store.trace, spec, 5)
