from copy import deepcopy

from conftest import injection_context
from driftguard.contracts.loader import load_openapi
from driftguard.runtime import CanonicalContractSnapshotManager


def test_alignment_rules_and_snapshot_isolation(injection_catalog):
    ae = injection_context(injection_catalog, "ICD-01", "AE")
    tf = injection_context(injection_catalog, "ICD-01", "TF")
    pd = injection_context(injection_catalog, "ICD-01", "PD")
    assert ae.contracts.contracts_equal
    assert tf.contracts.contracts_equal
    assert not pd.contracts.contracts_equal
    assert ae.contracts.alignment_mode == "aligned_new"
    assert tf.contracts.alignment_mode == "aligned_canonical"
    assert pd.contracts.alignment_mode == "displayed_old_runtime_new"
    canonical = deepcopy(tf.contracts.displayed.document)
    copy = tf.contracts.displayed.copy_document()
    copy["info"]["title"] = "mutated copy"
    assert tf.contracts.displayed.document == canonical


def test_pd_contract_changes_only_at_change_point(injection_catalog):
    context = injection_context(injection_catalog, "RSD-01", "PD", episode=2)
    assert context.contracts.contracts_equal
    context.set_episode(3)
    assert not context.contracts.contracts_equal
    assert context.contracts.displayed.source == "canonical_v1"


def test_snapshot_manager_hash_and_real_mutation_do_not_touch_canonical(injection_catalog):
    manager = CanonicalContractSnapshotManager()
    canonical = load_openapi()
    context = injection_context(injection_catalog, "ICD-02", "PD")
    runtime_schema = context.runtime_contract["components"]["schemas"]["TriggerPipelineRequest"]
    assert "branch" in runtime_schema["properties"] and "ref" not in runtime_schema["properties"]
    assert runtime_schema["required"] == ["branch"]
    assert load_openapi() == canonical
    assert manager.canonical_sha256 == "be893ee40a748410d49befe8ca45f645b47e9a770cf2bb7e0b90db9c71472267"


def test_newly_required_field_drops_the_old_default_in_mutated_snapshot(injection_catalog):
    context = injection_context(injection_catalog, "ICD-01", "AE")
    schema = context.displayed_contract["components"]["schemas"]["CreateIssueRequest"]
    assert "priority" in schema["required"]
    assert "default" not in schema["properties"]["priority"]
    canonical = load_openapi()["components"]["schemas"]["CreateIssueRequest"]
    assert canonical["properties"]["priority"]["default"] == "medium"


def test_scenario_reset_clears_sidecar_without_sharing_objects(injection_catalog):
    first = injection_context(injection_catalog, "SED-01", "PD")
    second = injection_context(injection_catalog, "SED-01", "PD")
    assert first.session is not second.session
    first.session.call_count = 9
    first.reset()
    assert first.session.call_count == 0
    assert second.session.call_count == 0
