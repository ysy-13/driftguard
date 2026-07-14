from __future__ import annotations

from copy import deepcopy
from typing import Callable

from driftguard.diagnosis.models import DiagnosisResult
from driftguard.evidence.models import AgentView, thaw
from driftguard.runtime.context import ExecutionContext

from .candidate_generator import PatchCandidateGenerator
from .minimality_validator import MinimalityValidator
from .models import GenerationResult, PatchLifecycle, ToolSpecPatch
from .overlay import SpecOverlay
from .patch_registry import PatchRegistry
from .regression_validator import RegressionValidator
from .repair_executor import RepairExecutor, repair_validation
from .safety_validator import SafetyValidator
from .static_validator import StaticPatchValidator


class HealingEngine:
    def __init__(self, max_candidates: int = 5, max_repair_calls: int = 8, registry: PatchRegistry | None = None):
        self.generator = PatchCandidateGenerator(max_candidates)
        self.static = StaticPatchValidator()
        self.executor = RepairExecutor(max_repair_calls)
        self.regression = RegressionValidator()
        self.safety = SafetyValidator()
        self.minimality = MinimalityValidator()
        self.registry = registry or PatchRegistry()
        self.max_repair_calls = max_repair_calls
        self.last_accepted_patch: ToolSpecPatch | None = None

    def heal(
        self,
        agent_view: AgentView,
        diagnosis: DiagnosisResult,
        context_factory: Callable[[], ExecutionContext] | None = None,
        repair_arguments: dict | None = None,
        initial_state: dict | None = None,
    ) -> dict:
        self.last_accepted_patch = None
        generation: GenerationResult = self.generator.generate(agent_view, diagnosis)
        base = {
            "disposition": generation.disposition,
            "candidate_count": len(generation.candidates),
            "reason_codes": list(generation.reason_codes),
            "candidate_summaries": [self._candidate_summary(candidate) for candidate in generation.candidates],
            "selected_patch": None,
            "accepted": False,
            "validation": {},
            "repair_run": None,
        }
        if not generation.candidates:
            return base
        if context_factory is None or repair_arguments is None:
            raise ValueError("eligible patches require a real persistent-runtime validation context")
        patch = generation.candidates[0]
        overlay = SpecOverlay(thaw(agent_view.displayed_spec))

        static_result, _ = self.static.validate(patch, thaw(agent_view.displayed_spec))
        base["validation"]["static"] = static_result.to_dict()
        if not static_result.passed:
            rejected = patch.with_status(PatchLifecycle.REJECTED, static_result)
            overlay.rollback()
            base["selected_patch"] = rejected.to_dict()
            return base
        patch = patch.with_status(PatchLifecycle.STATIC_VALIDATED, static_result)
        overlay.apply(patch)

        repair_run = self.executor.execute(patch, context_factory(), repair_arguments, initial_state)
        repair_result = repair_validation(repair_run)
        base["validation"]["immediate_repair"] = repair_result.to_dict()
        base["repair_run"] = repair_run.public_dict()
        if not repair_result.passed:
            rejected = patch.with_status(PatchLifecycle.REJECTED, repair_result)
            overlay.rollback()
            base["selected_patch"] = rejected.to_dict()
            return base
        patch = patch.with_status(PatchLifecycle.REPAIR_VALIDATED, repair_result)

        regression_result = self.regression.validate(patch, repair_run.initial_state)
        base["validation"]["regression"] = regression_result.to_dict()
        if not regression_result.passed:
            rejected = patch.with_status(PatchLifecycle.REJECTED, regression_result)
            overlay.rollback()
            base["selected_patch"] = rejected.to_dict()
            return base
        patch = patch.with_status(PatchLifecycle.REGRESSION_VALIDATED, regression_result)

        safety_result = self.safety.validate(patch, repair_run, self.max_repair_calls)
        base["validation"]["safety"] = safety_result.to_dict()
        if not safety_result.passed:
            rejected = patch.with_status(PatchLifecycle.REJECTED, safety_result)
            overlay.rollback()
            base["selected_patch"] = rejected.to_dict()
            return base
        patch = patch.with_status(PatchLifecycle.SAFETY_VALIDATED, safety_result)

        minimality_result = self.minimality.validate(patch)
        base["validation"]["minimality"] = minimality_result.to_dict()
        if not minimality_result.passed:
            rejected = patch.with_status(PatchLifecycle.REJECTED, minimality_result)
            overlay.rollback()
            base["selected_patch"] = rejected.to_dict()
            return base
        accepted = patch.with_status(PatchLifecycle.ACCEPTED, minimality_result)
        self.registry.register(accepted)
        self.last_accepted_patch = accepted
        base["selected_patch"] = accepted.to_dict()
        base["accepted"] = True
        return base

    @staticmethod
    def _candidate_summary(patch: ToolSpecPatch) -> dict:
        return {
            "patch_id": patch.patch_id,
            "target_tool_id": patch.target_tool_id,
            "drift_category": patch.drift_category,
            "location_path": patch.location_path,
            "operation_count": len(patch.openapi_operations),
            "evidence_refs": list(patch.evidence_refs),
            "lifecycle_status": patch.lifecycle_status.value,
        }
