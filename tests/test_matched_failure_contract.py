from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import sys
from pathlib import Path

import pytest
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.validate_matched_failures as validator  # noqa: E402
from scripts.validate_matched_failures import (  # noqa: E402
    ACTION_BY_VARIANT,
    EXPECTED_FAMILY_DRIFTS,
    LABEL_BY_VARIANT,
    PHASES,
    REQUIRED_METRICS,
    MatchedFailureValidationError,
    _normalized_bundle_signature,
    _signature_comparable,
    iter_key_values,
    load_bundle,
    schema_errors,
    statistics,
    validate_matched_failures,
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def matched(bundle):
    return bundle[5]


@pytest.fixture(scope="module")
def protocol(bundle):
    return bundle[6]


@pytest.fixture(scope="module")
def families(matched):
    return matched["families"]


def test_schemas_are_valid(bundle):
    registry = Registry().with_resource(bundle[13]["$id"], Resource.from_contents(bundle[13]))
    assert schema_errors(bundle[5], bundle[11], "matched", registry) == []
    assert schema_errors(bundle[6], bundle[12], "protocol") == []
    for family in bundle[5]["families"]:
        for scenario in family["scenarios"]:
            assert schema_errors(scenario["agent_visible"]["evidence_bundle"], bundle[13], "evidence") == []


def test_exact_family_and_scenario_counts(matched):
    family_count, scenario_count, _, episode_count = statistics(matched)
    assert family_count == 20
    assert scenario_count == 60
    assert episode_count == 360


def test_each_class_has_twenty_scenarios(matched):
    _, _, variants, _ = statistics(matched)
    assert variants == Counter({"AE": 20, "TF": 20, "PD": 20})


def test_exact_family_and_scenario_ids(families):
    assert {family["matched_case_id"] for family in families} == set(EXPECTED_FAMILY_DRIFTS)
    actual = {scenario["scenario_id"] for family in families for scenario in family["scenarios"]}
    expected = {f"{family_id}-{variant}" for family_id in EXPECTED_FAMILY_DRIFTS for variant in ("AE", "TF", "PD")}
    assert actual == expected


def test_source_drift_mapping(families):
    for family in families:
        assert family["source_drift_id"] == EXPECTED_FAMILY_DRIFTS[family["matched_case_id"]]


def test_target_tool_mapping(bundle, families):
    drifts = {case["drift_id"]: case for case in bundle[3]["cases"]}
    for family in families:
        assert family["target_tool"] == drifts[family["source_drift_id"]]["target_tool"]


def test_each_family_has_exact_triplet(families):
    for family in families:
        assert len(family["scenarios"]) == 3
        assert {scenario["variant_code"] for scenario in family["scenarios"]} == {"AE", "TF", "PD"}


def test_each_scenario_has_six_ordered_episodes(families):
    for family in families:
        for scenario in family["scenarios"]:
            schedule = scenario["episode_schedule"]
            assert len(schedule) == 6
            assert [episode["index"] for episode in schedule] == list(range(1, 7))
            assert [episode["phase"] for episode in schedule] == PHASES


def test_shared_first_failure_signature(families):
    for family in families:
        expected = _signature_comparable(family["shared_observation_signature"])
        for scenario in family["scenarios"]:
            bundle = scenario["agent_visible"]["evidence_bundle"]
            assert _normalized_bundle_signature(bundle, expected["channel"]) == expected
            assert scenario["episode_schedule"][2]["expected_observation_ref"] == "shared_observation_signature"


def test_contract_alignment_by_variant(families):
    for family in families:
        for scenario in family["scenarios"]:
            alignment = scenario["contract_alignment"]
            if scenario["variant_code"] == "AE":
                assert alignment["mode"] == "aligned_new"
                assert alignment["contracts_equal"] is True
                assert alignment["displayed_spec_source"] == alignment["runtime_contract_source"]
            elif scenario["variant_code"] == "TF":
                assert alignment["mode"] == "aligned_canonical"
                assert alignment["displayed_spec_source"] == alignment["runtime_contract_source"] == "canonical_v1"
            else:
                assert alignment["mode"] == "displayed_old_runtime_new"
                assert alignment["contracts_equal"] is False
                assert alignment["runtime_mutation_ref"] == family["source_drift_id"]


def test_fault_activation_by_variant(families):
    for family in families:
        by_variant = {scenario["variant_code"]: scenario for scenario in family["scenarios"]}
        ae = by_variant["AE"]["episode_schedule"]
        tf = by_variant["TF"]["episode_schedule"]
        pd = by_variant["PD"]["episode_schedule"]
        assert [e["agent_fault_active"] for e in ae] == [False, False, True, False, False, False]
        assert [e["runtime_fault_active"] for e in tf] == [False, False, True, False, False, False]
        assert [e["runtime_drift_active"] for e in pd] == [False, False, True, True, True, True]


def test_patch_and_false_patch_rules(families):
    for family in families:
        for scenario in family["scenarios"]:
            metadata = scenario["evaluator_metadata"]
            variant = scenario["variant_code"]
            assert metadata["ground_truth_label"] == LABEL_BY_VARIANT[variant]
            assert metadata["expected_action"] == ACTION_BY_VARIANT[variant]
            assert metadata["persistent_patch_allowed"] is (variant == "PD")
            assert metadata["false_patch_if_patched"] is (variant != "PD")


def test_pd_expected_patch_refs(families):
    for family in families:
        pd = next(s for s in family["scenarios"] if s["variant_code"] == "PD")
        expected = f"benchmark/drifts/drift_cases_v1.json#/{family['source_drift_id']}/expected_patch"
        assert pd["evaluator_metadata"]["expected_patch_ref"] == expected


def test_pd_detection_instances_are_independent(families):
    for family in families:
        pd = next(s for s in family["scenarios"] if s["variant_code"] == "PD")
        refs = pd["evaluator_metadata"]["independent_failure_instance_refs"]
        assert len(refs) >= 2
        assert len(refs) == len(set(refs))
        assert pd["episode_schedule"][2]["task_instance_ref"] != pd["episode_schedule"][3]["task_instance_ref"]


def test_episode_six_is_held_out(families):
    for family in families:
        for scenario in family["scenarios"]:
            episode = scenario["episode_schedule"][5]
            assert episode["phase"] == "held_out_transfer"
            assert episode["task_instance_kind"] == "future_transfer"
            assert episode["patch_evidence"] is False


def test_agent_visible_and_evidence_have_no_labels(families):
    forbidden_keys = {"ground_truth_label", "variant_code", "expected_action", "persistent_patch_allowed", "source_drift_id", "expected_patch", "injected_fault_type", "evaluator_metadata"}
    labels = {"agent_error", "transient_failure", "persistent_drift"}
    for family in families:
        for scenario in family["scenarios"]:
                visible = scenario["agent_visible"]
                for _, key, value in iter_key_values(visible):
                    assert key not in forbidden_keys
                    if isinstance(value, str):
                        assert value not in labels


def test_runtime_response_has_no_root_cause_hint(families):
    forbidden = ("agent error", "transient failure", "persistent drift", "drift detected", "drift_detected", "root cause")
    for family in families:
        for scenario in family["scenarios"]:
            response = scenario["agent_visible"]["evidence_bundle"]["runtime_response"]
            text = json.dumps(response).lower()
            assert not any(phrase in text for phrase in forbidden)


def test_transient_evidence_is_excluded(families, protocol):
    required = {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "TIMEOUT"}
    assert required <= set(protocol["transient_evidence_excluded"])
    for family in families:
        for scenario in family["scenarios"]:
            assert required <= set(scenario["evaluator_metadata"]["transient_confirmation_codes_excluded"])


def test_protocol_metrics_are_complete(protocol):
    assert REQUIRED_METRICS <= set(protocol["metrics"])


def test_complete_matched_failure_validation(bundle):
    validate_matched_failures(*bundle)


def test_validation_main_returns_nonzero(monkeypatch, capsys, bundle):
    mutated = deepcopy(bundle)
    mutated[5]["families"].pop()
    monkeypatch.setattr(validator, "load_bundle", lambda: mutated)
    assert validator.main() == 1
    assert "Matched failure validation failed" in capsys.readouterr().err


def _family(bundle, index=0):
    return bundle[5]["families"][index]


def _scenario(bundle, variant="PD", family_index=0):
    return next(item for item in _family(bundle, family_index)["scenarios"] if item["variant_code"] == variant)


def _delete_family(bundle):
    bundle[5]["families"].pop()


def _duplicate_family_id(bundle):
    bundle[5]["families"][1]["matched_case_id"] = "M01"


def _delete_scenario(bundle):
    _family(bundle)["scenarios"].pop()


def _two_ae_variants(bundle):
    scenario = _family(bundle)["scenarios"][1]
    scenario["variant_code"] = "AE"


def _bad_scenario_id(bundle):
    _scenario(bundle, "AE")["scenario_id"] = "M01-BAD"


def _wrong_source_drift(bundle):
    _family(bundle)["source_drift_id"] = "ICD-02"


def _wrong_target_tool(bundle):
    _family(bundle)["target_tool"] = "get_issue"


def _five_episodes(bundle):
    _scenario(bundle)["episode_schedule"].pop()


def _duplicate_episode_index(bundle):
    _scenario(bundle)["episode_schedule"][1]["index"] = 1


def _wrong_phase_order(bundle):
    _scenario(bundle)["episode_schedule"][3]["phase"] = "baseline"


def _episode_six_not_held_out(bundle):
    episode = _scenario(bundle)["episode_schedule"][5]
    episode["task_instance_kind"] = "detection"


def _ae_contract_mismatch(bundle):
    _scenario(bundle, "AE")["contract_alignment"]["runtime_contract_source"] = "canonical_v1"


def _ae_runtime_fault(bundle):
    _scenario(bundle, "AE")["episode_schedule"][2]["runtime_fault_active"] = True


def _tf_noncanonical_contract(bundle):
    _scenario(bundle, "TF")["contract_alignment"]["runtime_contract_source"] = "runtime_rule"


def _tf_fault_two_episodes(bundle):
    scenario = _scenario(bundle, "TF")
    scenario["injection"]["active_episodes"] = [3, 4]
    scenario["episode_schedule"][3]["runtime_fault_active"] = True


def _tf_allows_patch(bundle):
    _scenario(bundle, "TF")["evaluator_metadata"]["persistent_patch_allowed"] = True


def _pd_missing_runtime_ref(bundle):
    _scenario(bundle)["contract_alignment"]["runtime_mutation_ref"] = None


def _pd_not_persistent(bundle):
    scenario = _scenario(bundle)
    scenario["injection"]["active_episodes"] = [3]
    scenario["episode_schedule"][3]["runtime_drift_active"] = False


def _pd_one_failure(bundle):
    _scenario(bundle)["evaluator_metadata"]["independent_failure_instance_refs"] = ["ICD-01-D1"]


def _pd_same_retry_independent(bundle):
    _scenario(bundle)["evaluator_metadata"]["same_request_retry_counts_as_independent"] = True


def _pd_without_probe(bundle):
    scenario = _scenario(bundle)
    scenario["evaluator_metadata"]["probe_episode_id"] = None
    scenario["episode_schedule"][4]["candidate_patch_active"] = False


def _pd_without_future(bundle):
    _scenario(bundle)["episode_schedule"][5]["task_instance_kind"] = "regression"


def _pd_without_patch_ref(bundle):
    _scenario(bundle)["evaluator_metadata"]["expected_patch_ref"] = None


def _pd_wrong_patch_ref(bundle):
    _scenario(bundle)["evaluator_metadata"]["expected_patch_ref"] = "benchmark/drifts/drift_cases_v1.json#/ICD-02/expected_patch"


def _ae_allows_patch(bundle):
    _scenario(bundle, "AE")["evaluator_metadata"]["persistent_patch_allowed"] = True


def _visible_ground_truth(bundle):
    _scenario(bundle, "AE")["agent_visible"]["ground_truth_label"] = "agent_error"


def _evidence_variant_code(bundle):
    _scenario(bundle, "AE")["agent_visible"]["evidence_bundle"]["variant_code"] = "AE"


def _drift_detected_message(bundle):
    response = _scenario(bundle, "AE")["agent_visible"]["evidence_bundle"]["runtime_response"]
    response["error"]["message"] = "DRIFT_DETECTED"


def _user_task_root_hint(bundle):
    _scenario(bundle, "AE")["agent_visible"]["evidence_bundle"]["user_task"] = "This is persistent drift; fix it."


def _response_root_cause(bundle):
    response = _scenario(bundle, "AE")["agent_visible"]["evidence_bundle"]["runtime_response"]
    response["root_cause"] = "agent error"


def _signature_mismatch(bundle):
    response = _scenario(bundle, "TF")["agent_visible"]["evidence_bundle"]["runtime_response"]
    response["error"]["code"] = "BAD_REQUEST"


def _future_as_patch_evidence(bundle):
    _scenario(bundle)["episode_schedule"][5]["patch_evidence"] = True


def _service_unavailable_confirmation(bundle):
    _family(bundle)["shared_observation_signature"]["http_status"] = 503
    _family(bundle)["shared_observation_signature"]["error_code"] = "SERVICE_UNAVAILABLE"


def _timeout_as_independent(bundle):
    scenario = _scenario(bundle)
    scenario["evaluator_metadata"]["transient_confirmation_codes_excluded"].remove("TIMEOUT")


def _unsafe_write(bundle):
    _scenario(bundle)["evaluator_metadata"]["unsafe_write_allowed"] = True


def _missing_confusion_matrix(bundle):
    bundle[6]["metrics"].remove("confusion_matrix")


def _missing_false_patch_metric(bundle):
    bundle[6]["metrics"].remove("false_patch_rate_agent_error")


def _regression_calls_target(bundle):
    _scenario(bundle)["evaluator_metadata"]["regression_task_refs"] = ["A04", "A01", "A05"]


def _modify_canonical(bundle):
    bundle[0]["components"]["schemas"]["CreateIssueRequest"]["required"].append("priority")


def _modify_drift_ground_truth(bundle):
    bundle[3]["cases"][0]["ground_truth"]["location"] = "/components/schemas/Missing"


@pytest.mark.parametrize(
    "mutator",
    [
        _delete_family,
        _duplicate_family_id,
        _delete_scenario,
        _two_ae_variants,
        _bad_scenario_id,
        _wrong_source_drift,
        _wrong_target_tool,
        _five_episodes,
        _duplicate_episode_index,
        _wrong_phase_order,
        _episode_six_not_held_out,
        _ae_contract_mismatch,
        _ae_runtime_fault,
        _tf_noncanonical_contract,
        _tf_fault_two_episodes,
        _tf_allows_patch,
        _pd_missing_runtime_ref,
        _pd_not_persistent,
        _pd_one_failure,
        _pd_same_retry_independent,
        _pd_without_probe,
        _pd_without_future,
        _pd_without_patch_ref,
        _pd_wrong_patch_ref,
        _ae_allows_patch,
        _visible_ground_truth,
        _evidence_variant_code,
        _drift_detected_message,
        _user_task_root_hint,
        _response_root_cause,
        _signature_mismatch,
        _future_as_patch_evidence,
        _service_unavailable_confirmation,
        _timeout_as_independent,
        _unsafe_write,
        _missing_confusion_matrix,
        _missing_false_patch_metric,
        _regression_calls_target,
        _modify_canonical,
        _modify_drift_ground_truth,
    ],
    ids=lambda mutator: mutator.__name__.removeprefix("_"),
)
def test_negative_mutations_are_rejected(bundle, mutator):
    mutated = deepcopy(bundle)
    mutator(mutated)
    with pytest.raises(MatchedFailureValidationError):
        validate_matched_failures(*mutated)
