from __future__ import annotations

import re
from typing import Any


FORBIDDEN_KEYS = {
    "ground_truth_label", "evaluator_metadata", "variant", "variant_code",
    "source_drift_id", "expected_action", "expected_patch_ref", "patch_allowed",
    "runtime_profile", "runtime_contract", "mutation_operation", "persistent_drift",
    "transient_failure", "agent_error",
}
FORBIDDEN_VALUE_TERMS = {
    "ground_truth_label", "evaluator_metadata", "source_drift_id",
    "persistent_drift", "transient_failure", "agent_error", "drift_detected",
}
PUBLIC_ID_PATTERN = re.compile(r"(?:^|[-_])(AE|TF|PD)(?:$|[-_])", re.IGNORECASE)


class EvidenceLeakageError(ValueError):
    pass


def assert_public_id(value: str) -> None:
    if PUBLIC_ID_PATTERN.search(value):
        raise EvidenceLeakageError("public scenario ID contains a forbidden attribution suffix")


def assert_agent_visible(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_KEYS:
                raise EvidenceLeakageError(f"forbidden AgentView field at {path}.{key}")
            assert_agent_visible(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_agent_visible(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(term in lowered for term in FORBIDDEN_VALUE_TERMS):
            raise EvidenceLeakageError(f"forbidden AgentView content at {path}")
