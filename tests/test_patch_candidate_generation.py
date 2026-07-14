from dataclasses import replace

import pytest

from driftguard.diagnosis.models import PatchEligibilityDecision
from driftguard.evidence.models import AgentView, EvaluatorView
from driftguard.healing import HealingLeakageError, PatchCandidateGenerator
from driftguard.runtime import RuntimeProfile
from phase8_helpers import artifacts


@pytest.mark.parametrize("family_id", [f"M{index:02d}" for index in range(1, 21)])
def test_all_pd_categories_generate_one_evidence_grounded_candidate(family_id):
    value = artifacts(family_id)
    result = PatchCandidateGenerator().generate(value["agent_view"], value["diagnosis"])
    assert result.disposition == "PROPOSED"
    assert len(result.candidates) == 1
    patch = result.candidates[0]
    assert patch.target_tool_id == value["diagnosis"].localization.tool_id
    assert patch.location_path == value["diagnosis"].localization.location_path
    assert all(operation.evidence_refs for operation in patch.openapi_operations)


@pytest.mark.parametrize("variant", ["AE", "TF"])
def test_non_persistent_failures_never_generate_patch(variant):
    value = artifacts("M01", variant)
    result = PatchCandidateGenerator().generate(value["agent_view"], value["diagnosis"])
    assert result.disposition == "FORBIDDEN" and not result.candidates


def test_pd_without_eligibility_never_generates_patch():
    value = artifacts("M01")
    forbidden = replace(value["diagnosis"].patch_eligibility, decision="FORBIDDEN")
    diagnosis = replace(value["diagnosis"], patch_eligibility=forbidden)
    result = PatchCandidateGenerator().generate(value["agent_view"], diagnosis)
    assert result.disposition == "NOT_APPLICABLE" and not result.candidates


def test_agent_view_is_deeply_frozen():
    view = artifacts("M01")["agent_view"]
    with pytest.raises(TypeError):
        view.displayed_spec["info"] = {}


def test_generator_rejects_evaluator_view():
    value = artifacts("M01")
    with pytest.raises(HealingLeakageError):
        PatchCandidateGenerator().generate(EvaluatorView(value["agent_view"], {}), value["diagnosis"])


def test_generator_rejects_runtime_profile():
    value = artifacts("M01")
    runtime = RuntimeProfile("x", "create_issue", "input_contract", True, True, 3, None, "add_required", {})
    with pytest.raises(HealingLeakageError):
        PatchCandidateGenerator().generate(runtime, value["diagnosis"])


def test_episode_six_cannot_enter_generation():
    value = artifacts("M01")
    future_view = AgentView(value["agent_view"].trace, value["agent_view"].displayed_spec, 6)
    with pytest.raises(HealingLeakageError, match="future-transfer"):
        PatchCandidateGenerator().generate(future_view, value["diagnosis"])


def test_candidate_budget_is_bounded():
    with pytest.raises(ValueError, match="between 1 and 5"):
        PatchCandidateGenerator(6)

