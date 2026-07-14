from dataclasses import replace

import pytest

from driftguard.evidence.models import thaw
from driftguard.healing import PatchLifecycle, PatchOperation, PatchRegistry, SpecOverlay
from driftguard.healing.patch_document import PatchApplicationError
from phase8_helpers import artifacts, candidate


def test_overlay_applies_to_deep_copy_only():
    view = artifacts("M01")["agent_view"]
    source = thaw(view.displayed_spec)
    overlay = SpecOverlay(source)
    patched = overlay.apply(candidate("M01"))
    assert patched["components"]["schemas"]["CreateIssueRequest"]["required"] == ["title", "priority"]
    assert source["components"]["schemas"]["CreateIssueRequest"]["required"] == ["title"]


def test_source_fingerprint_mismatch_rejected():
    overlay = SpecOverlay(thaw(artifacts("M01")["agent_view"].displayed_spec))
    with pytest.raises(PatchApplicationError, match="fingerprint"):
        overlay.apply(replace(candidate(), source_spec_fingerprint="0" * 64))


def test_invalid_json_pointer_rejected():
    patch = candidate()
    bad = replace(patch, openapi_operations=(replace(patch.openapi_operations[0], path="/missing/parent/x"),))
    with pytest.raises(PatchApplicationError, match="missing JSON Pointer"):
        SpecOverlay(thaw(artifacts("M01")["agent_view"].displayed_spec)).apply(bad)


def test_patch_apply_is_atomic_and_rolls_back_partial_success():
    patch = candidate()
    first = patch.openapi_operations[0]
    second = PatchOperation("add", "/missing/parent/x", 1, first.evidence_refs, "INVALID")
    bad = replace(patch, openapi_operations=(first, second))
    source = thaw(artifacts("M01")["agent_view"].displayed_spec)
    overlay = SpecOverlay(source)
    with pytest.raises(PatchApplicationError):
        overlay.apply(bad)
    assert overlay.document == source


def test_overlay_explicit_rollback_discards_patch():
    source = thaw(artifacts("M01")["agent_view"].displayed_spec)
    overlay = SpecOverlay(source)
    overlay.apply(candidate())
    overlay.rollback()
    assert overlay.document == source


def test_registry_scope_and_acceptance_gate():
    registry = PatchRegistry()
    with pytest.raises(ValueError, match="accepted"):
        registry.register(candidate())
    accepted = candidate().with_status(PatchLifecycle.ACCEPTED)
    registry.register(accepted)
    assert registry.get(accepted.target_tool_id, accepted.source_spec_fingerprint, accepted.location_path) == accepted
    assert len(registry) == 1

