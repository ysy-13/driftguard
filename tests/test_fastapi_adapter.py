from __future__ import annotations

from fastapi.testclient import TestClient

from driftguard.api.app import create_app
from driftguard.contracts import ContractRegistry
from driftguard.sandbox import SandboxService


AUTH = {"Authorization": "Bearer agent_admin"}


def test_public_routes_match_canonical_method_path_and_operation_id():
    service = SandboxService()
    application = create_app(service)
    schema = application.openapi()
    actual = {
        (method, path, operation["operationId"])
        for path, path_item in schema["paths"].items()
        for method, operation in path_item.items()
    }
    expected = {
        (contract.method, contract.path, contract.operation_id)
        for contract in service.registry.contracts()
    }
    assert actual == expected
    assert len(actual) == 12
    assert "/internal/reset" not in schema["paths"]


def test_representative_get_post_and_error_statuses():
    client = TestClient(create_app())
    response = client.get("/repositories/R1", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["data"]["default_branch"] == "main"

    response = client.post(
        "/repositories/R1/issues",
        headers=AUTH,
        json={"title": "API issue"},
    )
    assert response.status_code == 201
    assert response.json()["data"]["issue_id"] == 105
    assert response.json()["data"]["priority"] == "medium"

    assert client.post("/internal/reset").status_code == 200
    response = client.post("/repositories/R1/issues", headers=AUTH, json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    response = client.get("/repositories/R1")
    assert response.status_code == 403
    response = client.get("/repositories/missing", headers=AUTH)
    assert response.status_code == 404


def test_unknown_actor_is_forbidden():
    client = TestClient(create_app())
    response = client.get(
        "/repositories/R1",
        headers={"Authorization": "Bearer unknown"},
    )
    assert response.status_code == 403


def test_http_and_in_process_results_are_identical():
    http_service = SandboxService()
    direct_service = SandboxService()
    client = TestClient(create_app(http_service))
    arguments = {"repo_id": "R1", "issue_id": 101, "assignee": "bob"}
    direct = direct_service.call_tool("assign_issue", arguments, "agent_admin")
    response = client.put(
        "/repositories/R1/issues/101/assignee",
        headers=AUTH,
        json={"assignee": "bob"},
    )
    assert response.status_code == direct.status_code
    assert response.json() == direct.payload
    assert http_service.store.snapshot() == direct_service.store.snapshot()


def test_invalid_json_is_bad_request():
    client = TestClient(create_app())
    response = client.post(
        "/repositories/R1/issues",
        headers={**AUTH, "Content-Type": "application/json"},
        content="{invalid",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"
