#!/usr/bin/env python3
"""Validate DriftGuard phase-3 persistent drift definitions and coverage."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_openapi import (  # noqa: E402
    ContractValidationError,
    iter_operations,
    validate_contract,
)


OPENAPI_PATH = ROOT / "benchmark" / "openapi" / "driftguard_openapi_v1.yaml"
TASKS_PATH = ROOT / "benchmark" / "tasks" / "tasks_v1.json"
FIXTURE_PATH = ROOT / "benchmark" / "fixtures" / "initial_state_v1.json"
DRIFTS_PATH = ROOT / "benchmark" / "drifts" / "drift_cases_v1.json"
COVERAGE_PATH = ROOT / "benchmark" / "drifts" / "task_drift_coverage_v1.json"
DRIFT_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "drift_case_schema_v1.json"
COVERAGE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "task_drift_coverage_schema_v1.json"

EXPECTED_DRIFT_IDS = {
    *(f"ICD-{number:02d}" for number in range(1, 6)),
    *(f"RSD-{number:02d}" for number in range(1, 6)),
    *(f"WPD-{number:02d}" for number in range(1, 6)),
    *(f"SED-{number:02d}" for number in range(1, 6)),
}
EXPECTED_TYPE_COUNTS = {
    "input_contract": 5,
    "response_shape": 5,
    "workflow_precondition": 5,
    "state_effect": 5,
}
EXPECTED_PREFIX_TYPE = {
    "ICD": "input_contract",
    "RSD": "response_shape",
    "WPD": "workflow_precondition",
    "SED": "state_effect",
}
EXPECTED_PATCH_TYPES = {
    "input_contract": "input_contract_patch",
    "response_shape": "response_mapping_patch",
    "workflow_precondition": "workflow_precondition_patch",
    "state_effect": "state_effect_patch",
}
ALLOWED_MUTATIONS = {
    "input_contract": {"add_required", "rename_input_field", "replace_enum_value"},
    "response_shape": {"rename_output_field", "map_output_path", "replace_output_path"},
    "workflow_precondition": {"add_prerequisite", "add_verification_binding", "add_transition_step", "restrict_initial_transition"},
    "state_effect": {"change_effect_timing", "add_postcondition_verification", "replace_state_effect", "change_resource_identity"},
}
FORBIDDEN_LATER_PHASE_KEYS = {"AE", "TF", "PD", "agent_error", "transient_failure", "matched_failure"}
TRANSIENT_CODES = {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "TIMEOUT"}


class DriftValidationError(Exception):
    """Raised when phase-3 drift data violates an invariant."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DriftValidationError(f"{path}: JSON parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise DriftValidationError(f"{path}: root must be an object")
    return value


def load_openapi(path: Path = OPENAPI_PATH) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise DriftValidationError(f"{path}: OpenAPI parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise DriftValidationError(f"{path}: OpenAPI root must be an object")
    return value


def load_bundle() -> tuple[dict[str, Any], ...]:
    return (
        load_openapi(),
        load_json(TASKS_PATH),
        load_json(FIXTURE_PATH),
        load_json(DRIFTS_PATH),
        load_json(COVERAGE_PATH),
        load_json(DRIFT_SCHEMA_PATH),
        load_json(COVERAGE_SCHEMA_PATH),
    )


def schema_errors(instance: Any, schema: dict[str, Any], label: str) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = []
    for item in sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path)):
        location = "/".join(str(part) for part in item.absolute_path) or "$"
        errors.append(f"{label}:{location}: {item.message}")
    return errors


def resolve_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise DriftValidationError(f"invalid JSON pointer: {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise DriftValidationError(f"unresolvable canonical pointer: {pointer}")
    return current


def iter_keys(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield f"{location}.{key}", key
            yield from iter_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_keys(child, f"{location}[{index}]")


def _canonical_fact_matches(pointer: str, expected: Any, actual: Any) -> bool:
    if expected == actual:
        return True
    pointer_name = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    if isinstance(expected, str) and expected == pointer_name:
        return True
    if isinstance(expected, list) and isinstance(actual, dict):
        return set(expected) <= set(actual)
    return False


def _task_tools(task: dict[str, Any]) -> set[str]:
    return {step.get("tool") for step in task.get("oracle_plan", []) if isinstance(step, dict)}


def _instance_signature(instance: dict[str, Any]) -> str:
    material = {
        "base_task_id": instance.get("base_task_id"),
        "instruction": instance.get("instruction"),
        "argument_overrides": instance.get("argument_overrides"),
        "fixture_overrides": instance.get("fixture_overrides"),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def validate_drifts(
    openapi: dict[str, Any],
    task_document: dict[str, Any],
    fixture: dict[str, Any],
    drift_document: dict[str, Any],
    coverage_document: dict[str, Any],
    drift_schema: dict[str, Any],
    coverage_schema: dict[str, Any],
) -> None:
    errors = schema_errors(drift_document, drift_schema, "drifts")
    errors.extend(schema_errors(coverage_document, coverage_schema, "coverage"))

    def error(location: str, message: str) -> None:
        errors.append(f"{location}: {message}")

    try:
        validate_contract(openapi)
    except ContractValidationError as exc:
        error("canonical", f"canonical OpenAPI was modified or invalid: {exc}")

    operations = {operation["operationId"]: operation for _, _, operation in iter_operations(openapi)}
    tasks = {
        task["task_id"]: task
        for task in task_document.get("tasks", [])
        if isinstance(task, dict) and "task_id" in task
    }
    if len(tasks) != 32:
        error("tasks", "canonical task set must still contain 32 unique tasks")
    if fixture.get("fixture_id") != "S0":
        error("fixture.fixture_id", "canonical fixture must remain S0")
    versions = {
        drift_document.get("canonical_contract_version"),
        task_document.get("canonical_contract_version"),
        fixture.get("canonical_contract_version"),
        *(
            operation.get("x-canonical-contract-version")
            for operation in operations.values()
        ),
    }
    if versions != {"1.0"}:
        error("canonical_contract_version", f"all sources must agree on 1.0, got {versions}")

    for location, key in iter_keys(drift_document):
        if key in FORBIDDEN_LATER_PHASE_KEYS:
            error(location, f"later-phase label {key!r} is forbidden")

    cases = drift_document.get("cases", [])
    drift_ids = [case.get("drift_id") for case in cases if isinstance(case, dict)]
    if len(cases) != 20:
        error("drifts.cases", f"expected exactly 20 cases, found {len(cases)}")
    if set(drift_ids) != EXPECTED_DRIFT_IDS:
        error("drifts.cases.drift_id", "exact ICD/RSD/WPD/SED 01-05 set is required")
    if len(drift_ids) != len(set(drift_ids)):
        error("drifts.cases.drift_id", "drift IDs must be unique")
    type_counts = Counter(case.get("drift_type") for case in cases if isinstance(case, dict))
    if {name: type_counts[name] for name in EXPECTED_TYPE_COUNTS} != EXPECTED_TYPE_COUNTS:
        error("drifts.cases.drift_type", f"expected {EXPECTED_TYPE_COUNTS}, got {dict(type_counts)}")

    by_drift_id = {
        case["drift_id"]: case
        for case in cases
        if isinstance(case, dict) and "drift_id" in case
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        drift_id = case.get("drift_id", f"index-{index}")
        prefix = f"drift {drift_id}"
        drift_type = case.get("drift_type")
        expected_type = EXPECTED_PREFIX_TYPE.get(str(drift_id).split("-", 1)[0])
        if drift_type != expected_type:
            error(f"{prefix}.drift_type", f"ID prefix requires {expected_type!r}")
        target_tool = case.get("target_tool")
        if target_tool not in operations:
            error(f"{prefix}.target_tool", f"unknown OpenAPI operationId {target_tool!r}")
        if case.get("persistence") != "persistent":
            error(f"{prefix}.persistence", "must be persistent")
        if case.get("change_point") != drift_document.get("change_point_default"):
            error(f"{prefix}.change_point", "must use the declared default change point")

        displayed = case.get("displayed_contract", {})
        mutation = case.get("runtime_mutation", {})
        target = mutation.get("target")
        try:
            canonical_value = resolve_pointer(openapi, target)
        except DriftValidationError as exc:
            error(f"{prefix}.runtime_mutation.target", str(exc))
            canonical_value = object()
        if displayed.get("target") != target or case.get("ground_truth", {}).get("location") != target:
            error(prefix, "displayed_contract, runtime_mutation, and ground_truth must identify the same canonical location")
        if not _canonical_fact_matches(target, displayed.get("canonical_value"), canonical_value):
            error(f"{prefix}.displayed_contract.canonical_value", "does not match canonical v1")
        if not _canonical_fact_matches(target, mutation.get("before"), canonical_value):
            error(f"{prefix}.runtime_mutation.before", "does not match canonical v1")
        if mutation.get("before") == mutation.get("after"):
            error(f"{prefix}.runtime_mutation.after", "must differ from before")
        if mutation.get("operation") not in ALLOWED_MUTATIONS.get(drift_type, set()):
            error(f"{prefix}.runtime_mutation.operation", f"operation is invalid for {drift_type}")

        if drift_type == "input_contract" and "Request" not in str(target):
            error(f"{prefix}.runtime_mutation.target", "input drift must target a request schema")
        if drift_type == "response_shape" and not any(
            marker in str(target) for marker in ("/Issue/", "/PipelineRun/", "/Member/", "/Repository/")
        ):
            error(f"{prefix}.runtime_mutation.target", "response drift must target a canonical output model")
        if drift_type == "workflow_precondition" and not (
            "x-preconditions" in str(target) or "AddMemberRequest" in str(target)
        ):
            error(f"{prefix}.runtime_mutation.target", "workflow drift must define a prerequisite or transition")
        if drift_type == "state_effect" and "x-state-effects" not in str(target):
            error(f"{prefix}.runtime_mutation.target", "state-effect drift must target canonical state effects")

        patch = case.get("expected_patch", {})
        if patch.get("patch_type") != EXPECTED_PATCH_TYPES.get(drift_type):
            error(f"{prefix}.expected_patch.patch_type", "does not match drift type")
        if patch.get("target_tool") != target_tool:
            error(f"{prefix}.expected_patch.target_tool", "must match case target_tool")
        patch_operations = patch.get("operations", [])
        if not patch_operations:
            error(f"{prefix}.expected_patch.operations", "must not be empty")
        for patch_index, patch_operation in enumerate(patch_operations):
            patch_target = patch_operation.get("target", "")
            if not str(patch_target).startswith("/adapter/"):
                error(f"{prefix}.expected_patch.operations[{patch_index}].target", "patch must target an adapter, never canonical v1")
            if patch_operation.get("before") == patch_operation.get("after"):
                error(f"{prefix}.expected_patch.operations[{patch_index}]", "patch must change a value")
        if patch_operations:
            first_patch = patch_operations[0]
            if drift_type == "response_shape":
                if first_patch.get("before") != mutation.get("after") or first_patch.get("after") != mutation.get("before"):
                    error(f"{prefix}.expected_patch.operations[0]", "response mapping must normalize runtime after back to canonical before")
            elif drift_type == "input_contract":
                if first_patch.get("before") != mutation.get("before") or first_patch.get("after") != mutation.get("after"):
                    error(f"{prefix}.expected_patch.operations[0]", "input patch must adapt canonical before to runtime after")
            elif drift_type in {"workflow_precondition", "state_effect"}:
                patch_before = first_patch.get("before")
                patch_after = first_patch.get("after")
                if not (
                    isinstance(patch_before, list)
                    and isinstance(patch_after, list)
                    and len(patch_after) > len(patch_before)
                ):
                    error(
                        f"{prefix}.expected_patch.operations[0]",
                        "workflow/state repair must extend the canonical flow with prerequisite or verification steps",
                    )

        requirements = patch.get("validation_requirements", {})
        evidence = case.get("minimum_evidence", {})
        if requirements.get("probe_required") is not True or evidence.get("probe_required") is not True:
            error(prefix, "probe_required must be true")
        if requirements.get("independent_failures", 0) < 2 or evidence.get("independent_failures", 0) < 2:
            error(prefix, "at least two independent failures are required")
        if requirements.get("regression_tasks_required", 0) < 3 or evidence.get("regression_tasks_required", 0) < 3:
            error(prefix, "at least three regression tasks are required")
        if requirements.get("unsafe_write_allowed") is not False:
            error(f"{prefix}.expected_patch.validation_requirements.unsafe_write_allowed", "must be false")
        if evidence.get("same_task_retry_insufficient") is not True:
            error(f"{prefix}.minimum_evidence.same_task_retry_insufficient", "must be true")
        excluded = set(evidence.get("transient_error_codes_excluded", []))
        if not {"RATE_LIMITED", "SERVICE_UNAVAILABLE"} <= excluded:
            error(f"{prefix}.minimum_evidence.transient_error_codes_excluded", "must exclude 429 and 503 evidence")
        symptom = case.get("observable_symptom", {})
        if symptom.get("http_status") in {429, 503} or symptom.get("error_code") in TRANSIENT_CODES:
            error(f"{prefix}.observable_symptom", "transient errors cannot confirm persistent drift")

        affected = case.get("affected_tasks", {})
        for task_id in affected.get("canonical_task_ids", []) + affected.get("control_task_ids", []):
            if task_id not in tasks:
                error(f"{prefix}.affected_tasks", f"unknown canonical task {task_id!r}")
        if drift_type == "workflow_precondition" and mutation.get("operation") not in ALLOWED_MUTATIONS["workflow_precondition"]:
            error(prefix, "workflow drift lacks a prerequisite relationship")
        if drift_type == "state_effect":
            state_patch_operations = {item.get("operation") for item in patch_operations}
            if not state_patch_operations & {"add_postcondition_verification", "change_resource_identity"}:
                error(prefix, "state-effect drift patch lacks postcondition verification or resource identity handling")

    coverage_entries = coverage_document.get("coverage", [])
    coverage_ids = [entry.get("drift_id") for entry in coverage_entries if isinstance(entry, dict)]
    if set(coverage_ids) != EXPECTED_DRIFT_IDS or len(coverage_entries) != 20:
        error("coverage", "must contain exactly one entry for every drift case")
    if len(coverage_ids) != len(set(coverage_ids)):
        error("coverage.drift_id", "coverage drift IDs must be unique")
    all_instance_ids: list[str] = []
    for index, entry in enumerate(coverage_entries):
        if not isinstance(entry, dict):
            continue
        drift_id = entry.get("drift_id", f"index-{index}")
        prefix = f"coverage {drift_id}"
        case = by_drift_id.get(drift_id)
        if case is None:
            error(f"{prefix}.drift_id", "coverage references an unknown drift case")
            continue
        target_tool = case.get("target_tool")
        if entry.get("target_tool") != target_tool:
            error(f"{prefix}.target_tool", "must match drift case")
        detection = entry.get("detection_instances", [])
        future = entry.get("future_transfer_instances", [])
        regression = entry.get("regression_tasks", [])
        if len(detection) < 2:
            error(f"{prefix}.detection_instances", "at least two are required")
        if len(future) < 1:
            error(f"{prefix}.future_transfer_instances", "at least one is required")
        if len(regression) < 3:
            error(f"{prefix}.regression_tasks", "at least three are required")
        detection_signatures = [_instance_signature(instance) for instance in detection]
        if len(detection_signatures) != len(set(detection_signatures)):
            error(f"{prefix}.detection_instances", "independent evidence cannot be identical retries")
        future_signatures = [_instance_signature(instance) for instance in future]
        if set(detection_signatures) & set(future_signatures):
            error(f"{prefix}.future_transfer_instances", "held-out instances must differ from detection evidence")
        for role, instances in (("detection", detection), ("future_transfer", future)):
            for instance_index, instance in enumerate(instances):
                location = f"{prefix}.{role}[{instance_index}]"
                instance_id = instance.get("instance_id")
                all_instance_ids.append(instance_id)
                if not str(instance_id).startswith(f"{drift_id}-"):
                    error(f"{location}.instance_id", "must be namespaced by drift ID")
                expected_held_out = role == "future_transfer"
                if instance.get("evaluation_role") != role:
                    error(f"{location}.evaluation_role", f"must be {role}")
                if instance.get("held_out") is not expected_held_out:
                    error(f"{location}.held_out", f"must be {expected_held_out}")
                if role == "future_transfer" and instance.get("patch_evidence") is not False:
                    error(f"{location}.patch_evidence", "held-out data cannot generate a patch")
                if role == "detection" and instance.get("patch_evidence") is not True:
                    error(f"{location}.patch_evidence", "detection data must be eligible independent evidence")
                base_task_id = instance.get("base_task_id")
                base_task = tasks.get(base_task_id)
                if base_task is None:
                    error(f"{location}.base_task_id", f"unknown canonical task {base_task_id!r}")
                elif target_tool not in _task_tools(base_task):
                    error(f"{location}.base_task_id", f"base task does not exercise {target_tool}")
                fixture_overrides = instance.get("fixture_overrides", {})
                if any(not str(pointer).startswith("/repositories/") for pointer in fixture_overrides):
                    error(f"{location}.fixture_overrides", "overrides must be instance-scoped repository pointers and cannot modify S0 metadata")
        for task_id in regression:
            task = tasks.get(task_id)
            if task is None:
                error(f"{prefix}.regression_tasks", f"unknown task {task_id!r}")
            elif target_tool in _task_tools(task):
                error(f"{prefix}.regression_tasks", f"task {task_id} invokes drifted tool {target_tool}")
        for task_id in entry.get("control_tasks", []):
            if task_id not in tasks:
                error(f"{prefix}.control_tasks", f"unknown task {task_id!r}")
    if len(all_instance_ids) != len(set(all_instance_ids)):
        error("coverage.instance_id", "instance IDs must be globally unique across detection and future transfer")

    if errors:
        raise DriftValidationError("\n".join(f"- {item}" for item in errors))


def statistics(coverage_document: dict[str, Any]) -> tuple[int, int, int]:
    entries = coverage_document.get("coverage", [])
    return (
        sum(len(entry.get("detection_instances", [])) for entry in entries),
        sum(len(entry.get("future_transfer_instances", [])) for entry in entries),
        sum(len(entry.get("regression_tasks", [])) for entry in entries),
    )


def main() -> int:
    try:
        bundle = load_bundle()
        validate_drifts(*bundle)
        detection, future, regression = statistics(bundle[4])
    except DriftValidationError as exc:
        print(f"Drift benchmark validation failed:\n{exc}", file=sys.stderr)
        return 1
    print("Drift benchmark validation passed.")
    print("Cases: 20")
    print("Input-contract: 5")
    print("Response-shape: 5")
    print("Workflow-precondition: 5")
    print("State-effect: 5")
    print("Coverage entries: 20")
    print(f"Detection instances: {detection}")
    print(f"Future-transfer instances: {future}")
    print(f"Regression mappings: {regression}")
    print("Canonical source unchanged: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
