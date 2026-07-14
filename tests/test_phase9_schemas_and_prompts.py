import hashlib
import json

import pytest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from driftguard.agents import AgentAction, LLMAttribution, LLMPatchProposal
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm.prompt_loader import PromptLoader
from phase8_helpers import candidate


SCHEMAS = PROJECT_ROOT / "benchmark" / "schemas"
PROMPTS = PROJECT_ROOT / "benchmark" / "prompts"


def load(name):
    return json.loads((SCHEMAS / name).read_text())


def test_agent_action_schema_and_model():
    value = {"action_type": "TOOL_CALL", "tool_id": "get_repository", "arguments": {"repo_id": "R1"}, "concise_decision_summary": "read"}
    Draft202012Validator(load("agent_action_schema_v1.json")).validate(value)
    assert AgentAction.from_dict(value).tool_id == "get_repository"


def test_llm_attribution_schema_model_and_evidence_validation():
    value = {"predicted_class": "PERSISTENT_DRIFT", "target_tool_id": "create_issue", "drift_category": "ICD", "location_type": "request", "location_path": "/x", "evidence_refs": ["EV1"], "requested_probe": None, "confidence": 0.8, "concise_reason": "repeated visible failure"}
    Draft202012Validator(load("llm_attribution_schema_v1.json")).validate(value)
    attribution = LLMAttribution.from_dict(value)
    attribution.validate_evidence({"EV1"})
    try:
        attribution.validate_evidence(set())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown evidence was accepted")


def test_llm_patch_proposal_requires_guard_and_real_toolspecpatch():
    attribution = LLMAttribution("PERSISTENT_DRIFT", "create_issue", "ICD", "request", "/x", ("EV1",), None, 0.8, "reason")
    proposal = LLMPatchProposal(candidate(), attribution)
    assert proposal.guard_requested and proposal.proposal.target_tool_id == "create_issue"


def test_llm_patch_proposal_schema_composes_attribution_and_toolspecpatch():
    patch_schema, attribution_schema = load("tool_spec_patch_schema_v1.json"), load("llm_attribution_schema_v1.json")
    registry = Registry().with_resource(
        "tool_spec_patch_schema_v1.json", Resource.from_contents(patch_schema)
    ).with_resource("llm_attribution_schema_v1.json", Resource.from_contents(attribution_schema))
    attribution = {"predicted_class": "PERSISTENT_DRIFT", "target_tool_id": "create_issue", "drift_category": "ICD", "location_type": "request", "location_path": "/x", "evidence_refs": ["EV1"], "requested_probe": None, "confidence": 0.8, "concise_reason": "reason"}
    Draft202012Validator(load("llm_patch_proposal_schema_v1.json"), registry=registry).validate({
        "proposal": candidate().to_dict(), "raw_attribution": attribution, "guard_requested": True,
    })


def test_experiment_record_and_manifest_schemas_are_draft_202012():
    for name in ("experiment_record_schema_v1.json", "experiment_manifest_schema_v1.json"):
        Draft202012Validator.check_schema(load(name))


def test_prompt_hash_is_stable_and_versioned():
    loader = PromptLoader(PROMPTS)
    first = loader.hash("base_tool_agent_v1.txt")
    second = hashlib.sha256(loader.load("base_tool_agent_v1.txt").encode()).hexdigest()
    assert first == second and len(first) == 64


def test_prompts_contain_no_ground_truth_or_benchmark_identifiers():
    forbidden = ("ground_truth", "expected_patch", "source_drift", "runtimeprofile", "M01", "ICD-01")
    for path in PROMPTS.glob("*.txt"):
        lowered = path.read_text().lower()
        assert not any(term.lower() in lowered for term in forbidden)


def test_baseline_prompts_share_one_unchanged_base():
    base = (PROMPTS / "base_tool_agent_v1.txt").read_text()
    assert base
    policy_texts = [(PROMPTS / name).read_text() for name in ("reflection_v1.txt", "validation_guided_v1.txt", "driftguard_attribution_v1.txt")]
    assert all(base not in policy for policy in policy_texts)
    assert len(set(policy_texts)) == 3


def test_real_smoke_example_contains_no_api_key():
    text = (PROJECT_ROOT / "configs/experiments/phase9_real_smoke.example.yaml").read_text()
    assert "DRIFTGUARD_LLM_API_KEY" not in text and "sk-" not in text


@pytest.mark.parametrize("canary", ["family M01", "case ICD-01", "source_drift_id", "ground_truth_label"])
def test_prompt_loader_rejects_identifier_and_hidden_label_canaries(tmp_path, canary):
    (tmp_path / "bad.txt").write_text(canary)
    with pytest.raises(ValueError):
        PromptLoader(tmp_path).load("bad.txt")
