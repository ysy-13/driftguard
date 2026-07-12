#!/usr/bin/env python3
"""Validate DriftGuard phase-2 fixture and canonical task definitions."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = ROOT / "benchmark" / "openapi" / "driftguard_openapi_v1.yaml"
FIXTURE_PATH = ROOT / "benchmark" / "fixtures" / "initial_state_v1.json"
TASKS_PATH = ROOT / "benchmark" / "tasks" / "tasks_v1.json"
FIXTURE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "initial_state_schema_v1.json"
TASK_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "task_schema_v1.json"

EXPECTED_TASK_IDS = {
    *(f"A{number:02d}" for number in range(1, 14)),
    *(f"C{number:02d}" for number in range(1, 12)),
    *(f"F{number:02d}" for number in range(1, 9)),
}
EXPECTED_SPLITS = {"calibration": 13, "detection": 11, "future_transfer": 8}
READ_TOOLS = {"get_repository", "get_issue", "get_pipeline_status", "get_member"}
CREATE_ISSUE_TOOL = "create_issue"
CREATE_PIPELINE_TOOL = "trigger_pipeline"
ROLE_TO_PERMISSION = {
    "read": "read",
    "triage": "read",
    "write": "write",
    "maintain": "write",
    "admin": "admin",
}
FORBIDDEN_TASK_KEYS = {
    "ground_truth_label", "drift_id", "expected_patch", "AE", "TF", "PD"
}
STEP_REF_PATTERN = re.compile(r"^\$steps\.([^.]+)(?:\..+)?$")
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


class TaskValidationError(Exception):
    """Raised when phase-2 benchmark data violates an invariant."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskValidationError(f"{path}: JSON parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskValidationError(f"{path}: document root must be an object")
    return value


def load_openapi(path: Path = OPENAPI_PATH) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise TaskValidationError(f"{path}: OpenAPI parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskValidationError(f"{path}: OpenAPI root must be an object")
    return value


def load_benchmark() -> tuple[dict[str, Any], ...]:
    return (
        load_openapi(),
        load_json(FIXTURE_PATH),
        load_json(TASKS_PATH),
        load_json(FIXTURE_SCHEMA_PATH),
        load_json(TASK_SCHEMA_PATH),
    )


def resolve_ref(document: dict[str, Any], value: Any) -> Any:
    seen: set[str] = set()
    while isinstance(value, dict) and set(value) == {"$ref"}:
        ref = value["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise TaskValidationError(f"unsupported reference: {ref!r}")
        if ref in seen:
            raise TaskValidationError(f"circular reference: {ref}")
        seen.add(ref)
        current: Any = document
        for raw_token in ref[2:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, dict) or token not in current:
                raise TaskValidationError(f"unresolvable reference: {ref}")
            current = current[token]
        value = current
    return value


def iter_operations(openapi: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for path, path_item in openapi.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                yield path, method.lower(), operation


def operation_map(openapi: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {operation["operationId"]: operation for _, _, operation in iter_operations(openapi)}


def _schema_errors(instance: Any, schema: dict[str, Any], label: str) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = []
    for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path)):
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{label}:{location}: {error.message}")
    return errors


def _iter_step_refs(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str) and value.startswith("$steps."):
        yield location, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_step_refs(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_step_refs(child, f"{location}[{index}]")


def _iter_keys(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield f"{location}.{key}", key
            yield from _iter_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keys(child, f"{location}[{index}]")


def _input_schemas(openapi: dict[str, Any], operation: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    schemas: dict[str, Any] = {}
    required: set[str] = set()
    for raw_parameter in operation.get("parameters", []):
        parameter = resolve_ref(openapi, raw_parameter)
        schemas[parameter["name"]] = resolve_ref(openapi, parameter["schema"])
        if parameter.get("required"):
            required.add(parameter["name"])
    request_body = operation.get("requestBody")
    if request_body:
        body_schema = resolve_ref(
            openapi, request_body["content"]["application/json"]["schema"]
        )
        for name, field_schema in body_schema.get("properties", {}).items():
            schemas[name] = resolve_ref(openapi, field_schema)
        required.update(body_schema.get("required", []))
    return schemas, required


def _is_literal_valid(value: Any, schema: dict[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "anyOf" in schema:
        return any(_is_literal_valid(value, variant) for variant in schema["anyOf"])
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        return any(_is_literal_valid(value, {**schema, "type": item}) for item in expected_type)
    type_ok = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "null": lambda item: item is None,
    }.get(expected_type, lambda _item: True)(value)
    if not type_ok:
        return False
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            return False
        if schema.get("pattern") and re.fullmatch(schema["pattern"], value) is None:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
    if isinstance(value, dict) and len(value) > schema.get("maxProperties", len(value)):
        return False
    return True


def _canonical_entity_errors(openapi: dict[str, Any], fixture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schemas = openapi["components"]["schemas"]
    mapping = {
        "Repository": (repo for repo in fixture["repositories"].values()),
        "Issue": (
            issue
            for repo in fixture["repositories"].values()
            for issue in repo["issues"].values()
        ),
        "PipelineRun": (
            run
            for repo in fixture["repositories"].values()
            for run in repo["pipeline_runs"].values()
        ),
        "Member": (
            member
            for repo in fixture["repositories"].values()
            for member in repo["members"].values()
        ),
    }
    for schema_name, entities in mapping.items():
        canonical_schema = resolve_ref(openapi, schemas[schema_name])
        canonical_fields = set(canonical_schema["properties"])
        for entity in entities:
            projection = {key: entity[key] for key in canonical_fields if key in entity}
            errors.extend(_schema_errors(projection, canonical_schema, f"fixture:{schema_name}"))
    return errors


def validate_benchmark(
    openapi: dict[str, Any],
    fixture: dict[str, Any],
    task_document: dict[str, Any],
    fixture_schema: dict[str, Any],
    task_schema: dict[str, Any],
) -> None:
    errors = _schema_errors(fixture, fixture_schema, "fixture")
    errors.extend(_schema_errors(task_document, task_schema, "tasks"))

    def error(location: str, message: str) -> None:
        errors.append(f"{location}: {message}")

    if fixture.get("fixture_id") != "S0" or task_document.get("fixture_id") != "S0":
        error("fixture_id", "fixture and tasks must both use S0")
    contract_version = fixture.get("canonical_contract_version")
    if contract_version != task_document.get("canonical_contract_version") or contract_version != "1.0":
        error("canonical_contract_version", "fixture, tasks, and canonical v1 must agree")

    tasks = task_document.get("tasks", [])
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, dict)]
    if len(tasks) != 32:
        error("tasks", f"expected exactly 32 tasks, found {len(tasks)}")
    if set(task_ids) != EXPECTED_TASK_IDS:
        error("tasks.task_id", f"expected exact A01-A13, C01-C11, F01-F08 set")
    if len(task_ids) != len(set(task_ids)):
        error("tasks.task_id", "task IDs must be unique")
    split_counts = Counter(task.get("split") for task in tasks if isinstance(task, dict))
    if {key: split_counts[key] for key in EXPECTED_SPLITS} != EXPECTED_SPLITS:
        error("tasks.split", f"expected {EXPECTED_SPLITS}, got {dict(split_counts)}")

    operations = operation_map(openapi)
    canonical_tools = set(operations)
    repositories = fixture.get("repositories", {})
    actors = fixture.get("actors", {})
    users = fixture.get("users", {})
    if set(repositories) != {"R1", "R2", "R3"}:
        error("fixture.repositories", "S0 must contain exactly R1, R2, and R3")
    if set(users) != {"alice", "bob", "carol", "dave", "erin", "frank"}:
        error("fixture.users", "S0 must contain exactly the six canonical users")
    operation_versions = {
        operation.get("x-canonical-contract-version") for operation in operations.values()
    }
    if operation_versions != {contract_version}:
        error("canonical_contract_version", "task and fixture version must match every OpenAPI operation")
    required_permissions = {
        operation.get("x-required-permission") for operation in operations.values()
    }
    actor_permissions = set(actors.get("agent_admin", {}).get("permissions", []))
    if not required_permissions <= actor_permissions:
        error("fixture.actors.agent_admin.permissions", "does not cover all canonical operations")

    for location, key in _iter_keys(task_document):
        if key in FORBIDDEN_TASK_KEYS:
            error(location, f"forbidden later-phase field {key!r}")

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id", "<unknown>")
        prefix = f"task {task_id}"
        if task.get("actor_id") not in actors:
            error(f"{prefix}.actor_id", "actor does not exist in S0")
        if task.get("initial_state") != "S0" or task.get("reset_after") is not True:
            error(prefix, "task must start from S0 and reset_after must be true")
        if task_id.startswith("F") and task.get("split") != "future_transfer":
            error(f"{prefix}.split", "F tasks must be future_transfer")

        plan = task.get("oracle_plan", [])
        step_ids = [step.get("step_id") for step in plan if isinstance(step, dict)]
        if len(step_ids) != len(set(step_ids)):
            error(f"{prefix}.oracle_plan", "step IDs must be unique")
        if task.get("max_tool_calls", 0) < len(plan):
            error(f"{prefix}.max_tool_calls", "must be at least the oracle step count")

        write_task = False
        seen_steps: set[str] = set()
        created_members: dict[str, set[str]] = {}
        created_issue_repos: set[str] = set()
        created_pipeline_repos: set[str] = set()
        for index, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            step_location = f"{prefix}.oracle_plan[{index}]"
            step_id = step.get("step_id")
            for ref_location, ref in _iter_step_refs(step):
                match = STEP_REF_PATTERN.fullmatch(ref)
                if match is None:
                    error(f"{step_location}{ref_location[1:]}", f"invalid step reference {ref!r}")
                elif match.group(1) not in seen_steps:
                    error(f"{step_location}{ref_location[1:]}", f"reference must target an earlier step: {ref}")

            tool = step.get("tool")
            operation = operations.get(tool)
            if operation is None:
                error(f"{step_location}.tool", f"tool {tool!r} is not an OpenAPI operationId")
                if step_id:
                    seen_steps.add(step_id)
                continue
            if operation.get("x-effect-type") == "write":
                write_task = True

            arguments = step.get("arguments", {})
            input_schemas, required_arguments = _input_schemas(openapi, operation)
            missing_arguments = required_arguments - set(arguments)
            if missing_arguments:
                error(f"{step_location}.arguments", f"missing required arguments {sorted(missing_arguments)}")
            unknown_arguments = set(arguments) - set(input_schemas)
            if unknown_arguments:
                error(f"{step_location}.arguments", f"unknown canonical arguments {sorted(unknown_arguments)}")
            for name, value in arguments.items():
                if name not in input_schemas or (isinstance(value, str) and value.startswith("$steps.")):
                    continue
                if not _is_literal_valid(value, input_schemas[name]):
                    error(f"{step_location}.arguments.{name}", f"literal {value!r} violates canonical input schema")

            repo_id = arguments.get("repo_id")
            repo = repositories.get(repo_id) if isinstance(repo_id, str) and not repo_id.startswith("$steps.") else None
            if isinstance(repo_id, str) and not repo_id.startswith("$steps.") and repo is None:
                error(f"{step_location}.arguments.repo_id", f"repository {repo_id!r} does not exist")
            if repo is not None:
                default_branch = arguments.get("default_branch")
                if tool == "update_repository" and isinstance(default_branch, str):
                    if default_branch not in repo["branches"]:
                        error(f"{step_location}.arguments.default_branch", f"branch {default_branch!r} does not exist in {repo_id}")
                issue_id = arguments.get("issue_id")
                if tool in {"get_issue", "assign_issue", "close_issue"} and isinstance(issue_id, int):
                    if str(issue_id) not in repo["issues"]:
                        error(f"{step_location}.arguments.issue_id", f"issue {issue_id} does not exist in {repo_id}")
                run_id = arguments.get("run_id")
                if tool in {"get_pipeline_status", "retry_pipeline"} and isinstance(run_id, int):
                    if str(run_id) not in repo["pipeline_runs"]:
                        error(f"{step_location}.arguments.run_id", f"pipeline run {run_id} does not exist in {repo_id}")
                username = arguments.get("username")
                if tool == "add_member" and isinstance(username, str):
                    if username not in users:
                        error(f"{step_location}.arguments.username", f"global user {username!r} does not exist")
                    created_members.setdefault(repo_id, set()).add(username)
                if tool in {"get_member", "update_member_role"} and isinstance(username, str):
                    if username not in repo["members"] and username not in created_members.get(repo_id, set()):
                        error(f"{step_location}.arguments.username", f"active member {username!r} does not exist in {repo_id}")
                assignee = arguments.get("assignee")
                if tool == "assign_issue" and isinstance(assignee, str):
                    if assignee not in repo["members"] and assignee not in created_members.get(repo_id, set()):
                        error(f"{step_location}.arguments.assignee", f"assignee {assignee!r} is not an active member")
                if tool == "trigger_pipeline":
                    workflow_id = arguments.get("workflow_id")
                    workflow = repo["workflows"].get(workflow_id)
                    if workflow is None:
                        error(f"{step_location}.arguments.workflow_id", f"workflow {workflow_id!r} does not exist")
                    elif workflow["state"] != "active":
                        error(f"{step_location}.arguments.workflow_id", f"workflow {workflow_id!r} is not active")
                    ref = arguments.get("ref")
                    if isinstance(ref, str) and not ref.startswith("$steps.") and ref not in repo["branches"]:
                        error(f"{step_location}.arguments.ref", f"branch/ref {ref!r} does not exist in {repo_id}")
                    created_pipeline_repos.add(repo_id)
                if tool == CREATE_ISSUE_TOOL:
                    created_issue_repos.add(repo_id)
            if step_id:
                seen_steps.add(step_id)

        success_assertions = task.get("success_assertions", [])
        answer_assertions = task.get("answer_assertions", [])
        forbidden_assertions = task.get("forbidden_assertions", [])
        if write_task and not success_assertions:
            error(f"{prefix}.success_assertions", "write task must have a state success assertion")
        if not write_task and not answer_assertions:
            error(f"{prefix}.answer_assertions", "read-only task must have an answer assertion")
        if not forbidden_assertions:
            error(f"{prefix}.forbidden_assertions", "every task requires unchanged_outside")
            allowed_paths: set[str] = set()
        else:
            allowed_paths = {
                path
                for assertion in forbidden_assertions
                if isinstance(assertion, dict)
                for path in assertion.get("allowed_paths", [])
            }
        if not write_task and allowed_paths:
            error(f"{prefix}.forbidden_assertions", "read-only task must have empty allowed_paths")
        success_paths = {
            assertion.get("path")
            for assertion in success_assertions
            if isinstance(assertion, dict)
        }
        for repo_id in created_issue_repos:
            next_id = fixture["next_ids"][repo_id]["issue_id"]
            entity_path = f"/repositories/{repo_id}/issues/{next_id}/**"
            next_path = f"/next_ids/{repo_id}/issue_id"
            if entity_path not in allowed_paths or next_path not in allowed_paths:
                error(f"{prefix}.forbidden_assertions", "create_issue must allow the new entity and issue next_id paths")
            if f"/repositories/{repo_id}/issues/{next_id}" not in success_paths:
                error(f"{prefix}.success_assertions", f"must assert creation of issue {next_id}")
        for repo_id in created_pipeline_repos:
            next_id = fixture["next_ids"][repo_id]["run_id"]
            entity_path = f"/repositories/{repo_id}/pipeline_runs/{next_id}/**"
            next_path = f"/next_ids/{repo_id}/run_id"
            if entity_path not in allowed_paths or next_path not in allowed_paths:
                error(f"{prefix}.forbidden_assertions", "trigger_pipeline must allow the new entity and run next_id paths")
            if f"/repositories/{repo_id}/pipeline_runs/{next_id}" not in success_paths:
                error(f"{prefix}.success_assertions", f"must assert creation of pipeline run {next_id}")

    for repo_key, repo in repositories.items():
        if repo.get("repo_id") != repo_key:
            error(f"fixture.repositories.{repo_key}.repo_id", "must match repository key")
        if repo.get("default_branch") not in repo.get("branches", []):
            error(f"fixture.repositories.{repo_key}.default_branch", "must exist in branches")
        for workflow_key, workflow in repo.get("workflows", {}).items():
            if workflow.get("workflow_id") != workflow_key:
                error(f"fixture.repositories.{repo_key}.workflows.{workflow_key}", "workflow_id must match key")
        for username, member in repo.get("members", {}).items():
            if member.get("username") != username or member.get("repo_id") != repo_key:
                error(f"fixture.repositories.{repo_key}.members.{username}", "member keys must match canonical identifiers")
            expected_permission = ROLE_TO_PERMISSION.get(member.get("role"))
            if member.get("base_permission") != expected_permission:
                error(f"fixture.repositories.{repo_key}.members.{username}.base_permission", "does not match role mapping")
        for issue_key, issue in repo.get("issues", {}).items():
            if str(issue.get("issue_id")) != issue_key or issue.get("repo_id") != repo_key:
                error(f"fixture.repositories.{repo_key}.issues.{issue_key}", "issue keys must match canonical identifiers")
            if issue.get("state") == "open" and (issue.get("closed_at") is not None or issue.get("resolution") is not None):
                error(f"fixture.repositories.{repo_key}.issues.{issue_key}", "open issue must have null resolution and closed_at")
            if issue.get("state") == "closed" and (issue.get("closed_at") is None or issue.get("resolution") is None):
                error(f"fixture.repositories.{repo_key}.issues.{issue_key}", "closed issue requires resolution and closed_at")
            assignee = issue.get("assignee")
            if assignee is not None and assignee not in repo.get("members", {}):
                error(f"fixture.repositories.{repo_key}.issues.{issue_key}.assignee", "must be an active repository member")
        for run_key, run in repo.get("pipeline_runs", {}).items():
            if str(run.get("run_id")) != run_key or run.get("repo_id") != repo_key:
                error(f"fixture.repositories.{repo_key}.pipeline_runs.{run_key}", "run keys must match canonical identifiers")
            if run.get("workflow_id") not in repo.get("workflows", {}):
                error(f"fixture.repositories.{repo_key}.pipeline_runs.{run_key}.workflow_id", "workflow does not exist")
            if run.get("ref") not in repo.get("branches", []):
                error(f"fixture.repositories.{repo_key}.pipeline_runs.{run_key}.ref", "ref does not exist in branches")
            if run.get("status") == "completed" and run.get("conclusion") is None:
                error(f"fixture.repositories.{repo_key}.pipeline_runs.{run_key}", "completed run requires a conclusion")
            if run.get("status") != "completed" and run.get("conclusion") is not None:
                error(f"fixture.repositories.{repo_key}.pipeline_runs.{run_key}", "non-completed run must have null conclusion")

    for repo_id, next_ids in fixture.get("next_ids", {}).items():
        repo = repositories.get(repo_id)
        if repo is None:
            error(f"fixture.next_ids.{repo_id}", "repository does not exist")
            continue
        issue_ids = [int(value) for value in repo["issues"]] or [next_ids["issue_id"] - 1]
        run_ids = [int(value) for value in repo["pipeline_runs"]] or [next_ids["run_id"] - 1]
        if next_ids["issue_id"] != max(issue_ids) + 1:
            error(f"fixture.next_ids.{repo_id}.issue_id", "must be one greater than the largest issue ID")
        if next_ids["run_id"] != max(run_ids) + 1:
            error(f"fixture.next_ids.{repo_id}.run_id", "must be one greater than the largest run ID")

    errors.extend(_canonical_entity_errors(openapi, fixture))
    if errors:
        raise TaskValidationError("\n".join(f"- {item}" for item in errors))


def main() -> int:
    try:
        validate_benchmark(*load_benchmark())
    except TaskValidationError as exc:
        print(f"Task benchmark validation failed:\n{exc}", file=sys.stderr)
        return 1
    print("Task benchmark validation passed.")
    print("Fixture: S0")
    print("Tasks: 32")
    print("Calibration: 13")
    print("Detection: 11")
    print("Future transfer: 8")
    print("Oracle tools: valid")
    print("Assertions: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
