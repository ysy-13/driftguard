from __future__ import annotations

from copy import deepcopy

import pytest

from conftest import service_from_state


def assert_atomic_failure(service, tool, arguments, actor="agent_admin", status=None):
    before = service.store.snapshot()
    clock = service.clock.now()
    result = service.call_tool(tool, arguments, actor)
    assert not result.ok
    if status is not None:
        assert result.status_code == status
    assert service.store.snapshot() == before
    assert service.clock.now() == clock
    assert service.call_log()[-1]["state_diff"] == []
    return result


def test_unknown_tool_and_input_failure_are_atomic(service):
    assert_atomic_failure(service, "missing_tool", {}, status=400)
    assert_atomic_failure(service, "create_issue", {"repo_id": "R1"}, status=422)


def test_permission_success_unknown_actor_and_missing_permission(service, initial_state):
    assert service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin").status_code == 200
    assert_atomic_failure(service, "get_repository", {"repo_id": "R1"}, "unknown", 403)
    state = deepcopy(initial_state)
    state["actors"]["reader"] = {"actor_id": "reader", "permissions": ["repository:read"]}
    limited = service_from_state(state)
    assert_atomic_failure(limited, "create_issue", {"repo_id": "R1", "title": "x"}, "reader", 403)


def test_repository_get_and_updates(service):
    get_result = service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin")
    assert get_result.payload["data"]["default_branch"] == "main"
    assert "branches" not in get_result.payload["data"]
    result = service.call_tool("update_repository", {"repo_id": "R1", "description": "new"}, "agent_admin")
    assert result.payload["data"]["description"] == "new"
    assert result.payload["data"]["default_branch"] == "main"
    result = service.call_tool("update_repository", {"repo_id": "R1", "default_branch": "develop"}, "agent_admin")
    assert result.payload["data"]["default_branch"] == "develop"


def test_repository_update_preconditions_are_atomic(service):
    assert_atomic_failure(service, "update_repository", {"repo_id": "R3", "description": "x"}, status=409)
    assert_atomic_failure(service, "update_repository", {"repo_id": "R1", "default_branch": "missing"}, status=409)
    assert_atomic_failure(service, "get_repository", {"repo_id": "missing"}, status=404)


def test_create_issue_defaults_and_explicit_priority(service):
    result = service.call_tool("create_issue", {"repo_id": "R1", "title": "default"}, "agent_admin")
    assert result.status_code == 201
    assert result.payload["data"]["issue_id"] == 105
    assert result.payload["data"]["priority"] == "medium"
    assert result.payload["data"]["created_at"] == "2026-01-01T00:00:01Z"
    service.reset()
    result = service.call_tool("create_issue", {"repo_id": "R1", "title": "high", "priority": "high"}, "agent_admin")
    assert result.payload["data"]["priority"] == "high"


def test_issue_failures_get_assign_and_close(service):
    assert_atomic_failure(service, "create_issue", {"repo_id": "R3", "title": "x"}, status=409)
    assert_atomic_failure(service, "get_issue", {"repo_id": "R1", "issue_id": 999}, status=404)
    result = service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "bob"}, "agent_admin")
    assert result.payload["data"]["assignee"] == "bob"
    service.reset()
    assert_atomic_failure(service, "assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "dave"}, status=409)
    result = service.call_tool("close_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin")
    assert result.payload["data"]["state"] == "closed"
    assert result.payload["data"]["resolution"] == "completed"
    assert result.payload["data"]["assignee"] is None
    assert_atomic_failure(service, "close_issue", {"repo_id": "R1", "issue_id": 101}, status=409)


def test_trigger_get_and_pipeline_preconditions(service):
    result = service.call_tool("trigger_pipeline", {"repo_id": "R1", "workflow_id": "ci", "ref": "release"}, "agent_admin")
    assert result.status_code == 201
    assert result.payload["data"]["run_id"] == 504
    assert result.payload["data"]["status"] == "queued"
    service.reset()
    assert_atomic_failure(service, "trigger_pipeline", {"repo_id": "R2", "workflow_id": "deploy", "ref": "main"}, status=409)
    assert_atomic_failure(service, "trigger_pipeline", {"repo_id": "R1", "workflow_id": "ci", "ref": "missing"}, status=409)
    assert_atomic_failure(service, "get_pipeline_status", {"repo_id": "R1", "run_id": 999}, status=404)


@pytest.mark.parametrize("repo_id,run_id", [("R1", 501), ("R2", 601)])
def test_retry_failed_or_cancelled_updates_original_run(service, repo_id, run_id):
    before_next = service.store.snapshot()["next_ids"][repo_id]["run_id"]
    result = service.call_tool("retry_pipeline", {"repo_id": repo_id, "run_id": run_id}, "agent_admin")
    assert result.status_code == 201
    assert result.payload["data"]["run_id"] == run_id
    assert result.payload["data"]["attempt"] == 2
    assert result.payload["data"]["retried_from"] == 1
    state = service.store.snapshot()
    assert state["next_ids"][repo_id]["run_id"] == before_next
    assert str(before_next) not in state["repositories"][repo_id]["pipeline_runs"]


def test_success_pipeline_cannot_retry(service):
    assert_atomic_failure(service, "retry_pipeline", {"repo_id": "R1", "run_id": 502}, status=409)


def test_membership_add_defaults_direct_triage_get_and_update(service):
    result = service.call_tool("add_member", {"repo_id": "R1", "username": "erin"}, "agent_admin")
    assert result.payload["data"]["role"] == "read"
    service.reset()
    result = service.call_tool("add_member", {"repo_id": "R1", "username": "erin", "role": "triage"}, "agent_admin")
    assert result.payload["data"]["membership_state"] == "active"
    assert result.payload["data"]["base_permission"] == "read"
    get_result = service.call_tool("get_member", {"repo_id": "R1", "username": "erin"}, "agent_admin")
    assert get_result.payload["data"]["role"] == "triage"
    update = service.call_tool("update_member_role", {"repo_id": "R1", "username": "erin", "role": "write"}, "agent_admin")
    assert update.payload["data"]["role"] == "write"
    assert update.payload["data"]["base_permission"] == "write"


def test_membership_failures_are_atomic(service):
    assert_atomic_failure(service, "add_member", {"repo_id": "R1", "username": "bob"}, status=409)
    assert_atomic_failure(service, "add_member", {"repo_id": "R1", "username": "nobody"}, status=404)
    assert_atomic_failure(service, "get_member", {"repo_id": "R1", "username": "erin"}, status=404)
    assert_atomic_failure(service, "update_member_role", {"repo_id": "R1", "username": "alice", "role": "read"}, status=409)


def test_read_calls_do_not_change_state_or_clock(service):
    before = service.store.snapshot()
    clock = service.clock.now()
    service.call_tool("get_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin")
    assert service.store.snapshot() == before
    assert service.clock.now() == clock
    assert service.call_log()[-1]["state_diff"] == []


def test_call_log_is_outside_business_state(service):
    before = service.store.snapshot()
    service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin")
    assert service.store.snapshot() == before
    assert len(service.call_log()) == 1
    assert "call_index" not in service.store.snapshot()


def test_canonical_v1_behavior_has_no_drift_semantics(service):
    issue = service.call_tool("create_issue", {"repo_id": "R1", "title": "canonical"}, "agent_admin")
    assert issue.payload["data"]["priority"] == "medium"
    assert "issue_id" in issue.payload["data"] and "id" not in issue.payload["data"]
    assert service.call_tool("trigger_pipeline", {"repo_id": "R1", "workflow_id": "ci", "branch": "main"}, "agent_admin").status_code == 422
    assert service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee_username": "bob"}, "agent_admin").status_code == 422

    pipeline = service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": 501}, "agent_admin")
    assert "status" in pipeline.payload["data"] and "conclusion" in pipeline.payload["data"]
    assert "state" not in pipeline.payload["data"] and "result" not in pipeline.payload["data"]
    member = service.call_tool("get_member", {"repo_id": "R1", "username": "bob"}, "agent_admin")
    assert "role" in member.payload["data"] and "base_permission" in member.payload["data"]
    repository = service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin")
    assert "default_branch" in repository.payload["data"] and "defaultBranch" not in repository.payload["data"]

    service.reset()
    close = service.call_tool("close_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin")
    assert close.ok and close.payload["data"]["state"] == "closed"
    assert service.call_tool("get_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin").payload["data"]["state"] == "closed"

    service.reset()
    promoted = service.call_tool("update_member_role", {"repo_id": "R1", "username": "dave", "role": "write"}, "agent_admin")
    assert promoted.ok and promoted.payload["data"]["role"] == "write"
    added = service.call_tool("add_member", {"repo_id": "R1", "username": "frank", "role": "maintain"}, "agent_admin")
    assert added.ok and added.payload["data"]["membership_state"] == "active"

    service.reset()
    next_run = service.store.snapshot()["next_ids"]["R1"]["run_id"]
    retried = service.call_tool("retry_pipeline", {"repo_id": "R1", "run_id": 501}, "agent_admin")
    assert retried.ok and retried.payload["data"]["run_id"] == 501
    assert service.store.snapshot()["next_ids"]["R1"]["run_id"] == next_run

    service.reset()
    updated = service.call_tool("update_repository", {"repo_id": "R1", "default_branch": "develop"}, "agent_admin")
    assert updated.ok
    assert service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin").payload["data"]["default_branch"] == "develop"
