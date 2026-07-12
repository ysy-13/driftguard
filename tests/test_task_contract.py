from __future__ import annotations

from collections import Counter
from copy import deepcopy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.validate_tasks as validator  # noqa: E402
from scripts.validate_tasks import (  # noqa: E402
    EXPECTED_SPLITS,
    EXPECTED_TASK_IDS,
    READ_TOOLS,
    ROLE_TO_PERMISSION,
    TaskValidationError,
    _canonical_entity_errors,
    _iter_keys,
    _iter_step_refs,
    _schema_errors,
    load_benchmark,
    operation_map,
    validate_benchmark,
)


@pytest.fixture(scope="module")
def bundle():
    return load_benchmark()


@pytest.fixture(scope="module")
def openapi(bundle):
    return bundle[0]


@pytest.fixture(scope="module")
def fixture(bundle):
    return bundle[1]


@pytest.fixture(scope="module")
def task_document(bundle):
    return bundle[2]


@pytest.fixture(scope="module")
def tasks(task_document):
    return task_document["tasks"]


@pytest.fixture(scope="module")
def by_id(tasks):
    return {task["task_id"]: task for task in tasks}


def test_fixture_json_schema(bundle):
    _, fixture, _, fixture_schema, _ = bundle
    assert _schema_errors(fixture, fixture_schema, "fixture") == []


def test_s0_expected_entities(fixture):
    assert fixture["fixture_id"] == "S0"
    assert fixture["clock"] == "2026-01-01T00:00:00Z"
    assert set(fixture["users"]) == {"alice", "bob", "carol", "dave", "erin", "frank"}
    assert set(fixture["actors"]) == {"agent_admin"}
    assert set(fixture["repositories"]) == {"R1", "R2", "R3"}
    assert fixture["next_ids"] == {
        "R1": {"issue_id": 105, "run_id": 504},
        "R2": {"issue_id": 202, "run_id": 602},
    }
    assert set(fixture["repositories"]["R1"]["issues"]) == {"101", "102", "103", "104"}
    assert set(fixture["repositories"]["R1"]["pipeline_runs"]) == {"501", "502", "503"}
    assert set(fixture["repositories"]["R2"]["issues"]) == {"201"}
    assert set(fixture["repositories"]["R2"]["pipeline_runs"]) == {"601"}
    assert fixture["repositories"]["R3"]["issues"] == {}
    assert fixture["repositories"]["R3"]["pipeline_runs"] == {}


def test_task_json_schema(bundle):
    _, _, task_document, _, task_schema = bundle
    assert _schema_errors(task_document, task_schema, "tasks") == []


def test_exact_task_count(tasks):
    assert len(tasks) == 32


def test_exact_task_ids(tasks):
    assert {task["task_id"] for task in tasks} == EXPECTED_TASK_IDS


def test_split_counts(tasks):
    assert Counter(task["split"] for task in tasks) == Counter(EXPECTED_SPLITS)


def test_task_ids_are_unique(tasks):
    task_ids = [task["task_id"] for task in tasks]
    assert len(task_ids) == len(set(task_ids))


def test_oracle_tools_come_from_real_openapi(openapi, tasks):
    canonical_tools = set(operation_map(openapi))
    assert canonical_tools
    for task in tasks:
        for step in task["oracle_plan"]:
            assert step["tool"] in canonical_tools


def test_step_ids_are_unique(tasks):
    for task in tasks:
        step_ids = [step["step_id"] for step in task["oracle_plan"]]
        assert len(step_ids) == len(set(step_ids)), task["task_id"]


def test_no_forward_step_references(tasks):
    for task in tasks:
        seen = set()
        for step in task["oracle_plan"]:
            for _, reference in _iter_step_refs(step):
                assert reference.split(".")[1] in seen, (task["task_id"], reference)
            seen.add(step["step_id"])


def test_max_tool_calls_cover_oracle_plan(tasks):
    for task in tasks:
        assert task["max_tool_calls"] >= len(task["oracle_plan"])


def test_read_tasks_have_answer_assertions(openapi, tasks):
    operations = operation_map(openapi)
    for task in tasks:
        read_only = all(operations[step["tool"]]["x-effect-type"] == "read" for step in task["oracle_plan"])
        if read_only:
            assert task["answer_assertions"], task["task_id"]


def test_write_tasks_have_state_assertions(openapi, tasks):
    operations = operation_map(openapi)
    for task in tasks:
        has_write = any(operations[step["tool"]]["x-effect-type"] == "write" for step in task["oracle_plan"])
        if has_write:
            assert task["success_assertions"], task["task_id"]
            assert all(item["target"] == "state" for item in task["success_assertions"])


def test_all_tasks_have_forbidden_assertion(tasks):
    for task in tasks:
        assert task["forbidden_assertions"]
        assert all(item["operator"] == "unchanged_outside" for item in task["forbidden_assertions"])


def test_read_only_tasks_allow_no_state_change(openapi, tasks):
    operations = operation_map(openapi)
    for task in tasks:
        read_only = all(operations[step["tool"]]["x-effect-type"] == "read" for step in task["oracle_plan"])
        if read_only:
            assert all(not item["allowed_paths"] for item in task["forbidden_assertions"])


def test_create_tasks_allow_next_id(fixture, tasks):
    for task in tasks:
        allowed = {
            path
            for assertion in task["forbidden_assertions"]
            for path in assertion["allowed_paths"]
        }
        for step in task["oracle_plan"]:
            repo_id = step["arguments"].get("repo_id")
            if step["tool"] == "create_issue":
                issue_id = fixture["next_ids"][repo_id]["issue_id"]
                assert f"/repositories/{repo_id}/issues/{issue_id}/**" in allowed
                assert f"/next_ids/{repo_id}/issue_id" in allowed
            if step["tool"] == "trigger_pipeline":
                run_id = fixture["next_ids"][repo_id]["run_id"]
                assert f"/repositories/{repo_id}/pipeline_runs/{run_id}/**" in allowed
                assert f"/next_ids/{repo_id}/run_id" in allowed


def test_fixture_entities_match_canonical_schemas(openapi, fixture):
    assert _canonical_entity_errors(openapi, fixture) == []


def test_role_base_permission_mapping(fixture):
    for repo in fixture["repositories"].values():
        for member in repo["members"].values():
            assert member["base_permission"] == ROLE_TO_PERMISSION[member["role"]]


def test_issue_state_is_consistent(fixture):
    for repo in fixture["repositories"].values():
        for issue in repo["issues"].values():
            if issue["state"] == "open":
                assert issue["closed_at"] is None
                assert issue["resolution"] is None
            else:
                assert issue["closed_at"] is not None
                assert issue["resolution"] in {"completed", "not_planned"}


def test_pipeline_state_is_consistent(fixture):
    for repo in fixture["repositories"].values():
        for run in repo["pipeline_runs"].values():
            if run["status"] == "completed":
                assert run["conclusion"] in {"success", "failure", "cancelled"}
            else:
                assert run["conclusion"] is None


def test_default_branch_exists(fixture):
    for repo in fixture["repositories"].values():
        assert repo["default_branch"] in repo["branches"]


def test_workflow_references_are_valid(fixture, tasks):
    for task in tasks:
        for step in task["oracle_plan"]:
            if step["tool"] != "trigger_pipeline":
                continue
            arguments = step["arguments"]
            workflow = fixture["repositories"][arguments["repo_id"]]["workflows"][arguments["workflow_id"]]
            assert workflow["state"] == "active"


def test_canonical_drift_fields_do_not_leak(fixture, task_document):
    forbidden_keys = {"drift_id", "expected_patch", "ground_truth_label", "AE", "TF", "PD"}
    assert not {key for _, key in _iter_keys(task_document) if key in forbidden_keys}
    serialized = repr((fixture, task_document))
    assert "assignee_username" not in serialized
    assert "developer" not in serialized
    assert "pending" not in serialized
    assert "run_attempt" not in serialized
    assert "defaultBranch" not in serialized


def test_future_transfer_not_detection(tasks):
    for task in tasks:
        if task["task_id"].startswith("F"):
            assert task["split"] == "future_transfer"


def test_complete_benchmark_validation(bundle):
    validate_benchmark(*bundle)


def test_validation_main_returns_nonzero(monkeypatch, capsys, bundle):
    mutated = deepcopy(bundle[2])
    mutated["tasks"].pop()
    monkeypatch.setattr(
        validator,
        "load_benchmark",
        lambda: (bundle[0], bundle[1], mutated, bundle[3], bundle[4]),
    )
    assert validator.main() == 1
    assert "Task benchmark validation failed" in capsys.readouterr().err


def _delete_task(bundle):
    bundle[2]["tasks"].pop()


def _duplicate_task_id(bundle):
    bundle[2]["tasks"][1]["task_id"] = "A01"


def _unknown_tool(bundle):
    bundle[2]["tasks"][0]["oracle_plan"][0]["tool"] = "missing_tool"


def _unknown_repo(bundle):
    bundle[2]["tasks"][0]["oracle_plan"][0]["arguments"]["repo_id"] = "R99"


def _unknown_issue(bundle):
    bundle[2]["tasks"][4]["oracle_plan"][0]["arguments"]["issue_id"] = 999


def _developer_role(bundle):
    bundle[2]["tasks"][10]["oracle_plan"][0]["arguments"]["role"] = "developer"


def _pending_membership(bundle):
    bundle[1]["repositories"]["R1"]["members"]["bob"]["membership_state"] = "pending"


def _unknown_branch(bundle):
    bundle[2]["tasks"][7]["oracle_plan"][0]["arguments"]["ref"] = "missing"


def _disabled_workflow(bundle):
    arguments = bundle[2]["tasks"][7]["oracle_plan"][0]["arguments"]
    arguments.update({"repo_id": "R2", "workflow_id": "deploy", "ref": "main"})


def _insufficient_tool_calls(bundle):
    bundle[2]["tasks"][15]["max_tool_calls"] = 2


def _forward_step_reference(bundle):
    bundle[2]["tasks"][13]["oracle_plan"][0]["arguments"]["title"] = "$steps.s2.data.title"


def _read_without_answer(bundle):
    bundle[2]["tasks"][0]["answer_assertions"] = []


def _write_without_success(bundle):
    bundle[2]["tasks"][1]["success_assertions"] = []


def _without_forbidden(bundle):
    bundle[2]["tasks"][1]["forbidden_assertions"] = []


def _create_without_next_id_permission(bundle):
    allowed = bundle[2]["tasks"][13]["forbidden_assertions"][0]["allowed_paths"]
    allowed.remove("/next_ids/R1/issue_id")


def _add_drift_id(bundle):
    bundle[2]["tasks"][0]["drift_id"] = "D01"


def _future_as_detection(bundle):
    bundle[2]["tasks"][24]["split"] = "detection"


def _open_issue_with_closed_at(bundle):
    bundle[1]["repositories"]["R1"]["issues"]["101"]["closed_at"] = "2026-01-01T00:00:00Z"


def _completed_without_conclusion(bundle):
    bundle[1]["repositories"]["R1"]["pipeline_runs"]["501"]["conclusion"] = None


def _missing_default_branch(bundle):
    bundle[1]["repositories"]["R1"]["default_branch"] = "missing"


@pytest.mark.parametrize(
    "mutator",
    [
        _delete_task,
        _duplicate_task_id,
        _unknown_tool,
        _unknown_repo,
        _unknown_issue,
        _developer_role,
        _pending_membership,
        _unknown_branch,
        _disabled_workflow,
        _insufficient_tool_calls,
        _forward_step_reference,
        _read_without_answer,
        _write_without_success,
        _without_forbidden,
        _create_without_next_id_permission,
        _add_drift_id,
        _future_as_detection,
        _open_issue_with_closed_at,
        _completed_without_conclusion,
        _missing_default_branch,
    ],
    ids=lambda mutator: mutator.__name__.removeprefix("_"),
)
def test_negative_mutations_are_rejected(bundle, mutator):
    mutated = deepcopy(bundle)
    mutator(mutated)
    with pytest.raises(TaskValidationError):
        validate_benchmark(*mutated)
