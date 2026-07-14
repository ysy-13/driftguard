from dataclasses import replace

from driftguard.evidence.models import thaw
from driftguard.healing import PatchLifecycle, PatchOperation, PatchRegistry
from driftguard.healing.leakage_guard import HealingLeakageError, assert_redacted
from driftguard.healing.minimality_validator import MinimalityValidator
from driftguard.healing.static_validator import StaticPatchValidator
from phase8_helpers import artifacts, candidate


def test_patched_openapi_remains_31_valid():
    patch = candidate("M06")
    result, document = StaticPatchValidator().validate(patch, thaw(artifacts("M06")["agent_view"].displayed_spec))
    assert result.passed and document["openapi"].startswith("3.1")


def test_static_validation_rejects_mismatch():
    patch = replace(candidate(), source_spec_fingerprint="f" * 64)
    result, document = StaticPatchValidator().validate(patch, thaw(artifacts("M01")["agent_view"].displayed_spec))
    assert not result.passed and document is None


def test_static_validation_rejects_single_unrelated_path():
    patch = candidate()
    unrelated = replace(patch.openapi_operations[0], path="/info/x-driftguard-unrelated")
    result, document = StaticPatchValidator().validate(
        replace(patch, openapi_operations=(unrelated,)), thaw(artifacts("M01")["agent_view"].displayed_spec)
    )
    assert not result.passed and "UNRELATED_TOOL_PATH" in result.reason_codes and document is None


def test_minimality_rejects_redundant_operation():
    patch = candidate()
    redundant = replace(patch, openapi_operations=patch.openapi_operations * 2)
    result = MinimalityValidator().validate(redundant)
    assert not result.passed and "REDUNDANT_OPERATION" in result.reason_codes


def test_minimality_rejects_unrelated_extra_path():
    patch = candidate()
    extra = PatchOperation("add", "/info/x-driftguard-extra", True, patch.evidence_refs, "UNRELATED")
    result = MinimalityValidator().validate(replace(patch, openapi_operations=patch.openapi_operations + (extra,)))
    assert not result.passed


def test_rejected_patch_does_not_enter_registry():
    registry = PatchRegistry()
    rejected = candidate().with_status(PatchLifecycle.REJECTED)
    try:
        registry.register(rejected)
    except ValueError:
        pass
    assert len(registry) == 0


def test_result_redaction_catches_ground_truth_and_tokens():
    for value in ({"ground_truth_label": "x"}, {"verification_token": "secret"}, {"variant": "PD"}):
        try:
            assert_redacted(value)
        except HealingLeakageError:
            continue
        raise AssertionError("leakage was not detected")


def test_result_redaction_accepts_public_patch():
    assert_redacted(candidate().to_dict())
