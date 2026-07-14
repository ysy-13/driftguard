from __future__ import annotations

import json
from typing import Any

from driftguard.diagnosis.models import DiagnosisResult
from driftguard.evidence.models import AgentView, EvaluatorView
from driftguard.runtime.execution_profile import RuntimeProfile


FORBIDDEN_RESULT_KEYS = {
    "ground_truth_label", "evaluator_metadata", "variant", "variant_code",
    "source_drift_id", "expected_patch_ref", "runtime_profile",
    "runtime_contract", "mutation_operation", "injection_strategy",
}


class HealingLeakageError(ValueError):
    pass


def validate_generator_inputs(agent_view: AgentView, diagnosis: DiagnosisResult) -> None:
    if isinstance(agent_view, EvaluatorView):
        raise HealingLeakageError("EvaluatorView is forbidden during patch generation")
    if isinstance(agent_view, RuntimeProfile) or isinstance(diagnosis, RuntimeProfile):
        raise HealingLeakageError("RuntimeProfile is forbidden during patch generation")
    if type(agent_view) is not AgentView or type(diagnosis) is not DiagnosisResult:
        raise TypeError("generator accepts only frozen AgentView and DiagnosisResult")
    if agent_view.current_episode > 5:
        raise HealingLeakageError("future-transfer evidence is forbidden during generation")


def assert_redacted(value: Any) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if str(key).lower() in FORBIDDEN_RESULT_KEYS or "token" in str(key).lower():
                    raise HealingLeakageError(f"forbidden result field at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")
    walk(value, "$")
    # Also catches accidentally serialized object representations.
    encoded = json.dumps(value, sort_keys=True).lower()
    for term in ("runtimeprofile", "hidden runtime", "expected_patch_ref"):
        if term in encoded:
            raise HealingLeakageError(f"forbidden serialized content: {term}")

