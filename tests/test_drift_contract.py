from __future__ import annotations

from collections import Counter
from copy import deepcopy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.validate_drifts as validator  # noqa: E402
from scripts.validate_drifts import (  # noqa: E402
    ALLOWED_MUTATIONS,
    EXPECTED_DRIFT_IDS,
    EXPECTED_PATCH_TYPES,
    EXPECTED_TYPE_COUNTS,
    DriftValidationError,
    _canonical_fact_matches,
    _task_tools,
    iter_keys,
    load_bundle,
    resolve_pointer,
    schema_errors,
    statistics,
    validate_drifts,
)
from scripts.validate_openapi import validate_contract  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def openapi(bundle):
    return bundle[0]


@pytest.fixture(scope="module")
def task_document(bundle):
    return bundle[1]


@pytest.fixture(scope="module")
def drift_document(bundle):
    return bundle[3]


@pytest.fixture(scope="module")
def coverage_document(bundle):
    return bundle[4]


@pytest.fixture(scope="module")
def cases(drift_document):
    return drift_document["cases"]


@pytest.fixture(scope="module")
def coverage(coverage_document):
    return coverage_document["coverage"]


def test_drift_json_schema(bundle):
    assert schema_errors(bundle[3], bundle[5], "drifts") == []


def test_coverage_json_schema(bundle):
    assert schema_errors(bundle[4], bundle[6], "coverage") == []


def test_exact_case_count(cases):
    assert len(cases) == 20


def test_exact_type_counts(cases):
    assert Counter(case["drift_type"] for case in cases) == Counter(EXPECTED_TYPE_COUNTS)


def test_exact_drift_ids(cases):
    assert {case["drift_id"] for case in cases} == EXPECTED_DRIFT_IDS


def test_drift_ids_are_unique(cases):
    ids = [case["drift_id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_target_tools_are_real_openapi_operations(openapi, cases):
    tools = {
        operation["operationId"]
        for _, _, operation in validator.iter_operations(openapi)
    }
    for case in cases:
        assert case["target_tool"] in tools


def test_target_paths_resolve(openapi, cases):
    for case in cases:
        assert resolve_pointer(openapi, case["runtime_mutation"]["target"]) is not None


def test_mutation_before_matches_canonical(openapi, cases):
    for case in cases:
        mutation = case["runtime_mutation"]
        canonical = resolve_pointer(openapi, mutation["target"])
        assert _canonical_fact_matches(mutation["target"], mutation["before"], canonical), case["drift_id"]
        displayed = case["displayed_contract"]
        assert _canonical_fact_matches(displayed["target"], displayed["canonical_value"], canonical)


def test_mutation_after_differs(cases):
    for case in cases:
        assert case["runtime_mutation"]["after"] != case["runtime_mutation"]["before"]


def test_expected_patch_direction(cases):
    for case in cases:
        mutation = case["runtime_mutation"]
        patch = case["expected_patch"]
        assert patch["patch_type"] == EXPECTED_PATCH_TYPES[case["drift_type"]]
        assert patch["target_tool"] == case["target_tool"]
        first = patch["operations"][0]
        assert first["target"].startswith("/adapter/")
        if case["drift_type"] == "response_shape":
            assert first["before"] == mutation["after"]
            assert first["after"] == mutation["before"]
        elif case["drift_type"] == "input_contract":
            assert first["before"] == mutation["before"]
            assert first["after"] == mutation["after"]
        else:
            assert isinstance(first["before"], list)
            assert isinstance(first["after"], list)
            assert len(first["after"]) > len(first["before"])


def test_minimum_evidence_is_complete(cases):
    for case in cases:
        evidence = case["minimum_evidence"]
        assert evidence["independent_failures"] >= 2
        assert evidence["probe_required"] is True
        assert evidence["regression_tasks_required"] >= 3
        assert evidence["same_task_retry_insufficient"] is True
        assert {"RATE_LIMITED", "SERVICE_UNAVAILABLE"} <= set(evidence["transient_error_codes_excluded"])


def test_every_case_has_coverage(cases, coverage):
    assert {case["drift_id"] for case in cases} == {entry["drift_id"] for entry in coverage}


def test_detection_future_and_regression_minimums(coverage):
    for entry in coverage:
        assert len(entry["detection_instances"]) >= 2
        assert len(entry["future_transfer_instances"]) >= 1
        assert len(entry["regression_tasks"]) >= 3


def test_held_out_rules(coverage):
    for entry in coverage:
        for instance in entry["detection_instances"]:
            assert instance["evaluation_role"] == "detection"
            assert instance["held_out"] is False
            assert instance["patch_evidence"] is True
        for instance in entry["future_transfer_instances"]:
            assert instance["evaluation_role"] == "future_transfer"
            assert instance["held_out"] is True
            assert instance["patch_evidence"] is False


def test_instance_ids_are_globally_unique(coverage):
    ids = [
        instance["instance_id"]
        for entry in coverage
        for group in (entry["detection_instances"], entry["future_transfer_instances"])
        for instance in group
    ]
    assert len(ids) == len(set(ids))


def test_base_tasks_exist_and_exercise_target(task_document, coverage):
    tasks = {task["task_id"]: task for task in task_document["tasks"]}
    for entry in coverage:
        for group in (entry["detection_instances"], entry["future_transfer_instances"]):
            for instance in group:
                assert instance["base_task_id"] in tasks
                assert entry["target_tool"] in _task_tools(tasks[instance["base_task_id"]])


def test_regression_tasks_do_not_call_target(task_document, coverage):
    tasks = {task["task_id"]: task for task in task_document["tasks"]}
    for entry in coverage:
        for task_id in entry["regression_tasks"]:
            assert task_id in tasks
            assert entry["target_tool"] not in _task_tools(tasks[task_id])


def test_canonical_openapi_is_unchanged(openapi):
    validate_contract(openapi)


def test_no_later_phase_labels(drift_document):
    forbidden = {"AE", "TF", "PD", "agent_error", "transient_failure", "matched_failure"}
    assert not {key for _, key in iter_keys(drift_document) if key in forbidden}


def test_workflow_drifts_have_prerequisite_relationship(cases):
    for case in cases:
        if case["drift_type"] == "workflow_precondition":
            assert case["runtime_mutation"]["operation"] in ALLOWED_MUTATIONS["workflow_precondition"]
            assert case["expected_patch"]["patch_type"] == "workflow_precondition_patch"


def test_state_effect_drifts_have_state_verification(cases):
    for case in cases:
        if case["drift_type"] == "state_effect":
            operations = {item["operation"] for item in case["expected_patch"]["operations"]}
            assert operations & {"add_postcondition_verification", "change_resource_identity"}


def test_patch_safety_requirements(cases):
    for case in cases:
        requirements = case["expected_patch"]["validation_requirements"]
        assert requirements == {
            "probe_required": True,
            "independent_failures": 2,
            "regression_tasks_required": 3,
            "unsafe_write_allowed": False,
        }


def test_coverage_statistics(coverage_document):
    assert statistics(coverage_document) == (40, 20, 60)


def test_complete_drift_validation(bundle):
    validate_drifts(*bundle)


def test_validation_main_returns_nonzero(monkeypatch, capsys, bundle):
    mutated = deepcopy(bundle)
    mutated[3]["cases"].pop()
    monkeypatch.setattr(validator, "load_bundle", lambda: mutated)
    assert validator.main() == 1
    assert "Drift benchmark validation failed" in capsys.readouterr().err


def _delete_case(bundle):
    bundle[3]["cases"].pop()


def _duplicate_drift_id(bundle):
    bundle[3]["cases"][1]["drift_id"] = "ICD-01"


def _unbalanced_types(bundle):
    bundle[3]["cases"][0]["drift_type"] = "response_shape"


def _unknown_target_tool(bundle):
    bundle[3]["cases"][0]["target_tool"] = "missing_tool"


def _missing_target_path(bundle):
    case = bundle[3]["cases"][0]
    for container, key in ((case, "displayed_contract"), (case, "runtime_mutation"), (case, "ground_truth")):
        container[key]["target" if key != "ground_truth" else "location"] = "/components/schemas/Missing/value"


def _before_mismatch(bundle):
    bundle[3]["cases"][0]["runtime_mutation"]["before"] = ["wrong"]


def _before_equals_after(bundle):
    mutation = bundle[3]["cases"][0]["runtime_mutation"]
    mutation["after"] = deepcopy(mutation["before"])


def _reverse_expected_patch(bundle):
    operation = bundle[3]["cases"][5]["expected_patch"]["operations"][0]
    operation["before"], operation["after"] = operation["after"], operation["before"]


def _one_independent_failure(bundle):
    bundle[3]["cases"][0]["minimum_evidence"]["independent_failures"] = 1


def _probe_not_required(bundle):
    bundle[3]["cases"][0]["minimum_evidence"]["probe_required"] = False


def _unsafe_write(bundle):
    bundle[3]["cases"][0]["expected_patch"]["validation_requirements"]["unsafe_write_allowed"] = True


def _delete_coverage(bundle):
    bundle[4]["coverage"].pop()


def _one_detection_instance(bundle):
    bundle[4]["coverage"][0]["detection_instances"].pop()


def _no_future_instance(bundle):
    bundle[4]["coverage"][0]["future_transfer_instances"] = []


def _two_regression_tasks(bundle):
    bundle[4]["coverage"][0]["regression_tasks"].pop()


def _future_not_held_out(bundle):
    bundle[4]["coverage"][0]["future_transfer_instances"][0]["held_out"] = False


def _detection_held_out(bundle):
    bundle[4]["coverage"][0]["detection_instances"][0]["held_out"] = True


def _unknown_base_task(bundle):
    bundle[4]["coverage"][0]["detection_instances"][0]["base_task_id"] = "A99"


def _regression_calls_target(bundle):
    bundle[4]["coverage"][0]["regression_tasks"] = ["A04", "A01", "A05"]


def _duplicate_detection_future_id(bundle):
    entry = bundle[4]["coverage"][0]
    entry["future_transfer_instances"][0]["instance_id"] = entry["detection_instances"][0]["instance_id"]


def _add_agent_error(bundle):
    bundle[3]["cases"][0]["agent_error"] = {"enabled": True}


def _add_transient_failure(bundle):
    bundle[3]["cases"][0]["transient_failure"] = {"enabled": True}


def _rate_limit_as_evidence(bundle):
    symptom = bundle[3]["cases"][0]["observable_symptom"]
    symptom["http_status"] = 429
    symptom["error_code"] = "RATE_LIMITED"


def _fixture_override_changes_s0(bundle):
    bundle[4]["coverage"][0]["future_transfer_instances"][0]["fixture_overrides"] = {"fixture_id": "S1"}


def _state_without_verification(bundle):
    patch = bundle[3]["cases"][15]["expected_patch"]["operations"][0]
    patch["operation"] = "change_effect_timing"
    patch["after"] = ["accept response without verification"]


def _workflow_without_prerequisite(bundle):
    bundle[3]["cases"][10]["runtime_mutation"]["operation"] = "add_required"


def _response_modifies_input(bundle):
    case = bundle[3]["cases"][5]
    target = "/components/schemas/CreateIssueRequest/required"
    case["displayed_contract"] = {"source": "canonical_v1", "target": target, "canonical_value": ["title"]}
    case["runtime_mutation"] = {"operation": "rename_output_field", "target": target, "before": ["title"], "after": ["title", "other"]}
    case["ground_truth"]["location"] = target
    patch = case["expected_patch"]["operations"][0]
    patch["before"], patch["after"] = ["title", "other"], ["title"]


def _input_modifies_output(bundle):
    case = bundle[3]["cases"][0]
    target = "/components/schemas/Issue/properties/issue_id"
    case["displayed_contract"] = {"source": "canonical_v1", "target": target, "canonical_value": "issue_id"}
    case["runtime_mutation"] = {"operation": "rename_input_field", "target": target, "before": "issue_id", "after": "id"}
    case["ground_truth"]["location"] = target
    patch = case["expected_patch"]["operations"][0]
    patch["before"], patch["after"] = "issue_id", "id"


def _patch_without_validation(bundle):
    bundle[3]["cases"][0]["expected_patch"].pop("validation_requirements")


def _canonical_directly_modified(bundle):
    bundle[0]["components"]["schemas"]["CreateIssueRequest"]["required"].append("priority")


@pytest.mark.parametrize(
    "mutator",
    [
        _delete_case,
        _duplicate_drift_id,
        _unbalanced_types,
        _unknown_target_tool,
        _missing_target_path,
        _before_mismatch,
        _before_equals_after,
        _reverse_expected_patch,
        _one_independent_failure,
        _probe_not_required,
        _unsafe_write,
        _delete_coverage,
        _one_detection_instance,
        _no_future_instance,
        _two_regression_tasks,
        _future_not_held_out,
        _detection_held_out,
        _unknown_base_task,
        _regression_calls_target,
        _duplicate_detection_future_id,
        _add_agent_error,
        _add_transient_failure,
        _rate_limit_as_evidence,
        _fixture_override_changes_s0,
        _state_without_verification,
        _workflow_without_prerequisite,
        _response_modifies_input,
        _input_modifies_output,
        _patch_without_validation,
        _canonical_directly_modified,
    ],
    ids=lambda mutator: mutator.__name__.removeprefix("_"),
)
def test_negative_mutations_are_rejected(bundle, mutator):
    mutated = deepcopy(bundle)
    mutator(mutated)
    with pytest.raises(DriftValidationError):
        validate_drifts(*mutated)
