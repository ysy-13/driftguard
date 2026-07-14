import pytest

from driftguard.evidence.leakage_guard import EvidenceLeakageError
from driftguard.experiments.provenance_audit import (
    assert_hidden_canary_blocked, assert_label_permutation_invariant,
    assert_opaque_id_invariant, audit_structured_patch_semantics,
)
from phase8_helpers import artifacts, candidate


def test_label_permutation_cannot_change_diagnosis_or_patch():
    view = artifacts("M01")["agent_view"]
    assert_label_permutation_invariant(view, [
        {"family": "permuted-A", "label": "TF"},
        {"family": "permuted-B", "label": "AE"},
    ])


def test_opaque_identifier_replacement_preserves_semantics():
    assert_opaque_id_invariant(artifacts("M01")["agent_view"])


@pytest.mark.parametrize("canary", [
    {"ground_truth_label": "CANARY"}, {"source_drift_id": "CANARY"},
    {"runtime_profile": "CANARY"}, {"expected_patch_ref": "CANARY"},
])
def test_hidden_label_canaries_are_blocked(canary):
    with pytest.raises(EvidenceLeakageError):
        assert_hidden_canary_blocked(canary)


@pytest.mark.parametrize("family_id", [f"M{index:02d}" for index in range(1, 21)])
def test_single_operation_patches_have_structured_validator_addressable_semantics(family_id):
    patch = candidate(family_id)
    assert len(patch.openapi_operations) == 1
    audit_structured_patch_semantics(patch)

