#!/usr/bin/env python3
"""Validate DriftGuard phase-4 matched failure triplets and attribution protocol."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_drifts import DriftValidationError, validate_drifts  # noqa: E402
from scripts.validate_openapi import ContractValidationError, validate_contract  # noqa: E402
from scripts.validate_tasks import TaskValidationError, validate_benchmark  # noqa: E402


OPENAPI_PATH = ROOT / "benchmark" / "openapi" / "driftguard_openapi_v1.yaml"
FIXTURE_PATH = ROOT / "benchmark" / "fixtures" / "initial_state_v1.json"
TASKS_PATH = ROOT / "benchmark" / "tasks" / "tasks_v1.json"
DRIFTS_PATH = ROOT / "benchmark" / "drifts" / "drift_cases_v1.json"
COVERAGE_PATH = ROOT / "benchmark" / "drifts" / "task_drift_coverage_v1.json"
MATCHED_PATH = ROOT / "benchmark" / "matched_failures" / "matched_failures_v1.json"
PROTOCOL_PATH = ROOT / "benchmark" / "matched_failures" / "attribution_protocol_v1.json"
FIXTURE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "initial_state_schema_v1.json"
TASK_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "task_schema_v1.json"
DRIFT_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "drift_case_schema_v1.json"
COVERAGE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "task_drift_coverage_schema_v1.json"
MATCHED_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "matched_failure_schema_v1.json"
PROTOCOL_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "attribution_protocol_schema_v1.json"
EVIDENCE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "evidence_bundle_schema_v1.json"

EXPECTED_FAMILY_DRIFTS = {
    **{f"M{number:02d}": f"ICD-{number:02d}" for number in range(1, 6)},
    **{f"M{number + 5:02d}": f"RSD-{number:02d}" for number in range(1, 6)},
    **{f"M{number + 10:02d}": f"WPD-{number:02d}" for number in range(1, 6)},
    **{f"M{number + 15:02d}": f"SED-{number:02d}" for number in range(1, 6)},
}
LABEL_BY_VARIANT = {"AE": "agent_error", "TF": "transient_failure", "PD": "persistent_drift"}
ACTION_BY_VARIANT = {
    "AE": "repair_current_call",
    "TF": "retry_or_recover_without_patch",
    "PD": "propose_validate_and_activate_patch",
}
PHASES = ["baseline", "baseline", "first_failure", "replication_or_recovery", "probe_or_confirmation", "held_out_transfer"]
REQUIRED_METRICS = {
    "attribution_accuracy", "macro_precision", "macro_recall", "macro_f1",
    "per_class_precision", "per_class_recall", "confusion_matrix",
    "false_patch_rate_agent_error", "false_patch_rate_transient_failure",
    "missed_drift_rate", "detection_delay",
}
TRANSIENT_EXCLUSIONS = {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "TIMEOUT"}
LEAKED_LABELS = {"agent_error", "transient_failure", "persistent_drift"}
FORBIDDEN_VISIBLE_KEYS = {
    "ground_truth_label", "variant_code", "expected_action",
    "persistent_patch_allowed", "source_drift_id", "expected_patch",
    "injected_fault_type", "evaluator_metadata",
}
FORBIDDEN_ROOT_CAUSE_PHRASES = {
    "agent error", "transient failure", "persistent drift", "stale specification",
    "specification changed", "drift detected", "drift_detected", "root cause",
}


class MatchedFailureValidationError(Exception):
    """Raised when phase-4 matched failure data violates an invariant."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedFailureValidationError(f"{path}: JSON parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedFailureValidationError(f"{path}: root must be an object")
    return value


def load_openapi(path: Path = OPENAPI_PATH) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise MatchedFailureValidationError(f"{path}: OpenAPI parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedFailureValidationError(f"{path}: OpenAPI root must be an object")
    return value


def load_bundle() -> tuple[dict[str, Any], ...]:
    return (
        load_openapi(),
        load_json(FIXTURE_PATH),
        load_json(TASKS_PATH),
        load_json(DRIFTS_PATH),
        load_json(COVERAGE_PATH),
        load_json(MATCHED_PATH),
        load_json(PROTOCOL_PATH),
        load_json(FIXTURE_SCHEMA_PATH),
        load_json(TASK_SCHEMA_PATH),
        load_json(DRIFT_SCHEMA_PATH),
        load_json(COVERAGE_SCHEMA_PATH),
        load_json(MATCHED_SCHEMA_PATH),
        load_json(PROTOCOL_SCHEMA_PATH),
        load_json(EVIDENCE_SCHEMA_PATH),
    )


def schema_errors(
    instance: Any,
    schema: dict[str, Any],
    label: str,
    registry: Registry | None = None,
) -> list[str]:
    kwargs = {"format_checker": FormatChecker()}
    if registry is not None:
        kwargs["registry"] = registry
    validator = Draft202012Validator(schema, **kwargs)
    errors = []
    for item in sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path)):
        location = "/".join(str(part) for part in item.absolute_path) or "$"
        errors.append(f"{label}:{location}: {item.message}")
    return errors


def iter_key_values(value: Any, location: str = "$") -> Iterator[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield child_location, key, child
            yield from iter_key_values(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_key_values(child, f"{location}[{index}]")


def _task_tools(task: dict[str, Any]) -> set[str]:
    return {step.get("tool") for step in task.get("oracle_plan", []) if isinstance(step, dict)}


def _normalized_bundle_signature(bundle: dict[str, Any], channel: str) -> dict[str, Any]:
    response = bundle.get("runtime_response", {})
    error = response.get("error") or {}
    result = {
        "channel": channel,
        "http_status": response.get("http_status"),
        "error_code": error.get("code"),
        "field": error.get("field") if error else response.get("missing_field"),
        "message_pattern": error.get("message") if error else response.get("message"),
        "response_success": response.get("success"),
    }
    state_diff = bundle.get("state_diff", [])
    if channel == "state_mismatch" and state_diff:
        result["expected_effect"] = state_diff[0].get("expected")
        result["observed_effect"] = state_diff[0].get("observed")
    return result


def _signature_comparable(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel": signature.get("channel"),
        "http_status": signature.get("http_status"),
        "error_code": signature.get("error_code"),
        "field": signature.get("field"),
        "message_pattern": signature.get("message_pattern"),
        "response_success": signature.get("response_success"),
        **(
            {"expected_effect": signature.get("expected_effect"), "observed_effect": signature.get("observed_effect")}
            if signature.get("channel") == "state_mismatch"
            else {}
        ),
    }


def validate_matched_failures(
    openapi: dict[str, Any],
    fixture: dict[str, Any],
    task_document: dict[str, Any],
    drift_document: dict[str, Any],
    coverage_document: dict[str, Any],
    matched_document: dict[str, Any],
    protocol: dict[str, Any],
    fixture_schema: dict[str, Any],
    task_schema: dict[str, Any],
    drift_schema: dict[str, Any],
    coverage_schema: dict[str, Any],
    matched_schema: dict[str, Any],
    protocol_schema: dict[str, Any],
    evidence_schema: dict[str, Any],
) -> None:
    evidence_registry = Registry().with_resource(
        evidence_schema["$id"], Resource.from_contents(evidence_schema)
    )
    errors = schema_errors(matched_document, matched_schema, "matched", evidence_registry)
    errors.extend(schema_errors(protocol, protocol_schema, "protocol"))

    def error(location: str, message: str) -> None:
        errors.append(f"{location}: {message}")

    try:
        validate_contract(openapi)
    except ContractValidationError as exc:
        error("canonical", f"OpenAPI modified or invalid: {exc}")
    try:
        validate_benchmark(openapi, fixture, task_document, fixture_schema, task_schema)
    except TaskValidationError as exc:
        error("phase2", f"fixture/tasks modified or invalid: {exc}")
    try:
        validate_drifts(
            openapi, task_document, fixture, drift_document, coverage_document,
            drift_schema, coverage_schema,
        )
    except DriftValidationError as exc:
        error("phase3", f"drift data modified or invalid: {exc}")

    families = matched_document.get("families", [])
    if len(families) != 20:
        error("matched.families", f"expected exactly 20 families, found {len(families)}")
    family_ids = [family.get("matched_case_id") for family in families if isinstance(family, dict)]
    if set(family_ids) != set(EXPECTED_FAMILY_DRIFTS):
        error("matched.families.matched_case_id", "exact M01-M20 set is required")
    if len(family_ids) != len(set(family_ids)):
        error("matched.families.matched_case_id", "family IDs must be unique")

    drifts = {case["drift_id"]: case for case in drift_document.get("cases", [])}
    coverage = {entry["drift_id"]: entry for entry in coverage_document.get("coverage", [])}
    tasks = {task["task_id"]: task for task in task_document.get("tasks", [])}
    scenario_ids: list[str] = []
    scenario_count = 0

    for family_index, family in enumerate(families):
        if not isinstance(family, dict):
            continue
        family_id = family.get("matched_case_id", f"index-{family_index}")
        prefix = f"family {family_id}"
        source_drift_id = family.get("source_drift_id")
        expected_drift_id = EXPECTED_FAMILY_DRIFTS.get(family_id)
        if source_drift_id != expected_drift_id:
            error(f"{prefix}.source_drift_id", f"expected {expected_drift_id!r}")
        drift = drifts.get(source_drift_id)
        if drift is None:
            error(f"{prefix}.source_drift_id", "source drift does not exist")
            continue
        if family.get("target_tool") != drift.get("target_tool"):
            error(f"{prefix}.target_tool", "must match source drift")
        entry = coverage.get(source_drift_id)
        if entry is None:
            error(prefix, "source drift coverage is missing")
            continue
        detection_ids = [item["instance_id"] for item in entry["detection_instances"]]
        future_ids = [item["instance_id"] for item in entry["future_transfer_instances"]]
        regression_ids = entry["regression_tasks"]
        future_by_id = {item["instance_id"]: item for item in entry["future_transfer_instances"]}
        signature = family.get("shared_observation_signature", {})
        visible_signature = _signature_comparable(signature)
        signature_message = str(signature.get("message_pattern", "")).lower()
        if any(phrase in signature_message for phrase in FORBIDDEN_ROOT_CAUSE_PHRASES):
            error(f"{prefix}.shared_observation_signature", "first-failure message leaks root cause")
        if signature.get("http_status") in {429, 503} or signature.get("error_code") in TRANSIENT_EXCLUSIONS:
            error(f"{prefix}.shared_observation_signature", "transient errors cannot serve as drift-confirming matched signatures")

        scenarios = family.get("scenarios", [])
        scenario_count += len(scenarios)
        variants = [scenario.get("variant_code") for scenario in scenarios if isinstance(scenario, dict)]
        if Counter(variants) != Counter({"AE": 1, "TF": 1, "PD": 1}):
            error(f"{prefix}.scenarios", "must contain exactly one AE, one TF, and one PD")
        for scenario_index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                continue
            variant = scenario.get("variant_code")
            scenario_id = scenario.get("scenario_id", f"index-{scenario_index}")
            scenario_prefix = f"scenario {scenario_id}"
            scenario_ids.append(scenario_id)
            if scenario_id != f"{family_id}-{variant}":
                error(f"{scenario_prefix}.scenario_id", "must combine family ID and variant code")

            metadata = scenario.get("evaluator_metadata", {})
            if metadata.get("ground_truth_label") != LABEL_BY_VARIANT.get(variant):
                error(f"{scenario_prefix}.evaluator_metadata.ground_truth_label", "does not match variant")
            if metadata.get("source_drift_id") != source_drift_id:
                error(f"{scenario_prefix}.evaluator_metadata.source_drift_id", "must match family source drift")
            if metadata.get("expected_action") != ACTION_BY_VARIANT.get(variant):
                error(f"{scenario_prefix}.evaluator_metadata.expected_action", "does not match variant")
            patch_allowed = variant == "PD"
            if metadata.get("persistent_patch_allowed") is not patch_allowed:
                error(f"{scenario_prefix}.evaluator_metadata.persistent_patch_allowed", f"must be {patch_allowed}")
            if metadata.get("false_patch_if_patched") is not (not patch_allowed):
                error(f"{scenario_prefix}.evaluator_metadata.false_patch_if_patched", "does not match patch policy")
            expected_patch_ref = f"benchmark/drifts/drift_cases_v1.json#/{source_drift_id}/expected_patch"
            if variant == "PD":
                if metadata.get("expected_patch_ref") != expected_patch_ref:
                    error(f"{scenario_prefix}.evaluator_metadata.expected_patch_ref", "must reference source drift expected patch")
                independent = metadata.get("independent_failure_instance_refs", [])
                if len(independent) < 2 or len(set(independent)) < 2:
                    error(f"{scenario_prefix}.evaluator_metadata.independent_failure_instance_refs", "PD needs two independent failures")
                if not set(independent) <= set(detection_ids):
                    error(f"{scenario_prefix}.evaluator_metadata.independent_failure_instance_refs", "must use source coverage detection instances")
                if metadata.get("probe_episode_id") != f"{scenario_id}-E5":
                    error(f"{scenario_prefix}.evaluator_metadata.probe_episode_id", "PD probe must be Episode 5")
            else:
                if metadata.get("expected_patch_ref") is not None:
                    error(f"{scenario_prefix}.evaluator_metadata.expected_patch_ref", "AE/TF cannot reference a persistent patch")
            if len(metadata.get("regression_task_refs", [])) < 3:
                error(f"{scenario_prefix}.evaluator_metadata.regression_task_refs", "at least three regressions are required")
            for task_id in metadata.get("regression_task_refs", []):
                task = tasks.get(task_id)
                if task is None or family.get("target_tool") in _task_tools(task):
                    error(f"{scenario_prefix}.evaluator_metadata.regression_task_refs", f"invalid target-free regression {task_id!r}")
            future_ref = metadata.get("future_transfer_instance_ref")
            if future_ref not in future_by_id or future_by_id.get(future_ref, {}).get("held_out") is not True:
                error(f"{scenario_prefix}.evaluator_metadata.future_transfer_instance_ref", "must reference held-out source coverage")
            if metadata.get("unsafe_write_allowed") is not False:
                error(f"{scenario_prefix}.evaluator_metadata.unsafe_write_allowed", "must be false")
            if metadata.get("same_request_retry_counts_as_independent") is not False:
                error(f"{scenario_prefix}.evaluator_metadata.same_request_retry_counts_as_independent", "same request retry is not independent")
            if not TRANSIENT_EXCLUSIONS <= set(metadata.get("transient_confirmation_codes_excluded", [])):
                error(f"{scenario_prefix}.evaluator_metadata.transient_confirmation_codes_excluded", "must exclude 429, 503, and timeout")

            alignment = scenario.get("contract_alignment", {})
            injection = scenario.get("injection", {})
            if variant == "AE":
                if not (
                    alignment.get("mode") == "aligned_new"
                    and alignment.get("contracts_equal") is True
                    and alignment.get("displayed_spec_source") == alignment.get("runtime_contract_source")
                    and alignment.get("displayed_mutation_ref") == source_drift_id
                    and alignment.get("runtime_mutation_ref") == source_drift_id
                ):
                    error(f"{scenario_prefix}.contract_alignment", "AE must align displayed and runtime on the new rule")
                expected_layer, expected_active, expected_persistent = "agent_behavior", [3], False
            elif variant == "TF":
                if not (
                    alignment.get("mode") == "aligned_canonical"
                    and alignment.get("contracts_equal") is True
                    and alignment.get("displayed_spec_source") == "canonical_v1"
                    and alignment.get("runtime_contract_source") == "canonical_v1"
                    and alignment.get("runtime_mutation_ref") is None
                ):
                    error(f"{scenario_prefix}.contract_alignment", "TF must align displayed and runtime on canonical v1")
                expected_layer, expected_active, expected_persistent = "runtime_once", [3], False
            else:
                if not (
                    alignment.get("mode") == "displayed_old_runtime_new"
                    and alignment.get("contracts_equal") is False
                    and alignment.get("displayed_spec_source") == "canonical_v1"
                    and alignment.get("runtime_mutation_ref") == source_drift_id
                    and alignment.get("change_point") == 3
                    and alignment.get("persistent") is True
                ):
                    error(f"{scenario_prefix}.contract_alignment", "PD must keep displayed canonical and reference persistent runtime drift")
                expected_layer, expected_active, expected_persistent = "runtime_contract", [3, 4, 5, 6], True
            if injection.get("fault_layer") != expected_layer:
                error(f"{scenario_prefix}.injection.fault_layer", f"must be {expected_layer}")
            if injection.get("active_episodes") != expected_active:
                error(f"{scenario_prefix}.injection.active_episodes", f"must be {expected_active}")
            if injection.get("persistent") is not expected_persistent:
                error(f"{scenario_prefix}.injection.persistent", f"must be {expected_persistent}")
            if injection.get("same_request_retry_counts_as_independent") is not False:
                error(f"{scenario_prefix}.injection", "same retries cannot count as independent")
            if not TRANSIENT_EXCLUSIONS <= set(injection.get("confirmation_excludes", [])):
                error(f"{scenario_prefix}.injection.confirmation_excludes", "must exclude transient evidence")

            schedule = scenario.get("episode_schedule", [])
            if len(schedule) != 6:
                error(f"{scenario_prefix}.episode_schedule", "must contain exactly six episodes")
                continue
            if [episode.get("index") for episode in schedule] != list(range(1, 7)):
                error(f"{scenario_prefix}.episode_schedule.index", "must be continuous 1-6")
            if [episode.get("phase") for episode in schedule] != PHASES:
                error(f"{scenario_prefix}.episode_schedule.phase", "phase order is invalid")
            for index, episode in enumerate(schedule, 1):
                if episode.get("episode_id") != f"{scenario_id}-E{index}":
                    error(f"{scenario_prefix}.episode_schedule[{index - 1}].episode_id", "must match scenario and index")
                if episode.get("visible_to_diagnoser") is not True:
                    error(f"{scenario_prefix}.episode_schedule[{index - 1}]", "episode evidence must be visible")
            episode3 = schedule[2]
            if episode3.get("expected_observation_ref") != "shared_observation_signature":
                error(f"{scenario_prefix}.episode_schedule[2]", "Episode 3 must reference the shared signature")
            episode6 = schedule[5]
            if not (
                episode6.get("phase") == "held_out_transfer"
                and episode6.get("task_instance_kind") == "future_transfer"
                and episode6.get("task_instance_ref") in future_ids
                and episode6.get("patch_evidence") is False
            ):
                error(f"{scenario_prefix}.episode_schedule[5]", "Episode 6 must be held-out and excluded from patch evidence")
            if variant == "AE":
                if [episode.get("agent_fault_active") for episode in schedule] != [False, False, True, False, False, False]:
                    error(f"{scenario_prefix}.episode_schedule", "AE must activate only the Episode 3 agent fault")
                if any(episode.get("runtime_fault_active") or episode.get("runtime_drift_active") or episode.get("candidate_patch_active") for episode in schedule):
                    error(f"{scenario_prefix}.episode_schedule", "AE cannot activate runtime faults, drift, or patches")
            elif variant == "TF":
                if [episode.get("runtime_fault_active") for episode in schedule] != [False, False, True, False, False, False]:
                    error(f"{scenario_prefix}.episode_schedule", "TF runtime fault must occur exactly once in Episode 3")
                if any(episode.get("agent_fault_active") or episode.get("runtime_drift_active") or episode.get("candidate_patch_active") for episode in schedule):
                    error(f"{scenario_prefix}.episode_schedule", "TF cannot activate agent faults, drift, or patches")
            else:
                if [episode.get("runtime_drift_active") for episode in schedule] != [False, False, True, True, True, True]:
                    error(f"{scenario_prefix}.episode_schedule", "PD must persist from Episode 3 through 6")
                if schedule[2].get("task_instance_ref") == schedule[3].get("task_instance_ref"):
                    error(f"{scenario_prefix}.episode_schedule", "PD Episodes 3 and 4 must use independent detection instances")
                if schedule[2].get("task_instance_ref") not in detection_ids or schedule[3].get("task_instance_ref") not in detection_ids:
                    error(f"{scenario_prefix}.episode_schedule", "PD failures must reference source detection coverage")
                if not (
                    schedule[4].get("task_instance_kind") == "probe"
                    and schedule[4].get("candidate_patch_active") is True
                    and schedule[5].get("candidate_patch_active") is True
                ):
                    error(f"{scenario_prefix}.episode_schedule", "PD needs Episode 5 probe and Episode 6 validated patch transfer")

            agent_visible = scenario.get("agent_visible", {})
            evidence = agent_visible.get("evidence_bundle", {})
            errors.extend(schema_errors(evidence, evidence_schema, f"{scenario_prefix}.evidence_bundle"))
            for location, key, value in iter_key_values(agent_visible, f"{scenario_prefix}.agent_visible"):
                if key in FORBIDDEN_VISIBLE_KEYS:
                    error(location, f"evaluator-only field {key!r} leaked into agent-visible evidence")
                if isinstance(value, str) and value in LEAKED_LABELS:
                    error(location, "ground-truth label leaked into agent-visible evidence")
            visible_text = json.dumps(agent_visible, sort_keys=True).lower()
            for phrase in FORBIDDEN_ROOT_CAUSE_PHRASES:
                if phrase in visible_text:
                    error(f"{scenario_prefix}.agent_visible", f"root-cause hint leaked: {phrase!r}")
            runtime_text = json.dumps(evidence.get("runtime_response", {}), sort_keys=True).lower()
            if any(phrase in runtime_text for phrase in FORBIDDEN_ROOT_CAUSE_PHRASES):
                error(f"{scenario_prefix}.agent_visible.evidence_bundle.runtime_response", "runtime response leaks root cause")
            if _normalized_bundle_signature(evidence, signature.get("channel")) != visible_signature:
                error(f"{scenario_prefix}.agent_visible.evidence_bundle", "Episode 3 evidence does not match family shared signature")

    if scenario_count != 60:
        error("matched.scenarios", f"expected 60 scenarios, found {scenario_count}")
    if len(scenario_ids) != len(set(scenario_ids)):
        error("matched.scenarios.scenario_id", "scenario IDs must be globally unique")
    expected_scenarios = {
        f"{family_id}-{variant}" for family_id in EXPECTED_FAMILY_DRIFTS for variant in LABEL_BY_VARIANT
    }
    if set(scenario_ids) != expected_scenarios:
        error("matched.scenarios.scenario_id", "exact M01-M20 AE/TF/PD scenario set is required")

    if protocol.get("labels") != ["agent_error", "transient_failure", "persistent_drift"]:
        error("protocol.labels", "label order and set must be canonical")
    if protocol.get("decision_point") != 5:
        error("protocol.decision_point", "must be Episode 5")
    if protocol.get("minimum_independent_drift_evidence", 0) < 2:
        error("protocol.minimum_independent_drift_evidence", "must be at least two")
    if protocol.get("same_request_retry_counts_as_independent") is not False:
        error("protocol.same_request_retry_counts_as_independent", "must be false")
    if not TRANSIENT_EXCLUSIONS <= set(protocol.get("transient_evidence_excluded", [])):
        error("protocol.transient_evidence_excluded", "must exclude 429, 503, and timeout")
    policies = protocol.get("patch_policy", {})
    for label in ("agent_error", "transient_failure"):
        policy = policies.get(label, {})
        if policy.get("persistent_patch_allowed") is not False or policy.get("patch_is_false_positive") is not True:
            error(f"protocol.patch_policy.{label}", "persistent patches must be forbidden and counted as false patches")
    pd_policy = policies.get("persistent_drift", {})
    if pd_policy.get("persistent_patch_allowed") is not True or pd_policy.get("patch_is_false_positive") is not False:
        error("protocol.patch_policy.persistent_drift", "validated persistent patch must be allowed")
    required_activation = {
        "generated_call_conforms_to_displayed_spec", "two_independent_failures",
        "exclude_transient_errors", "patch_explains_all_failures", "probe_succeeds",
        "three_regression_tasks_pass", "unsafe_write_rate_zero",
    }
    if set(pd_policy.get("activation_requirements", [])) != required_activation:
        error("protocol.patch_policy.persistent_drift.activation_requirements", "activation safeguards are incomplete")
    if not REQUIRED_METRICS <= set(protocol.get("metrics", [])):
        error("protocol.metrics", f"missing required metrics {sorted(REQUIRED_METRICS - set(protocol.get('metrics', [])))}")

    if errors:
        raise MatchedFailureValidationError("\n".join(f"- {item}" for item in errors))


def statistics(matched_document: dict[str, Any]) -> tuple[int, int, Counter[str], int]:
    families = matched_document.get("families", [])
    scenarios = [scenario for family in families for scenario in family.get("scenarios", [])]
    return (
        len(families),
        len(scenarios),
        Counter(scenario.get("variant_code") for scenario in scenarios),
        sum(len(scenario.get("episode_schedule", [])) for scenario in scenarios),
    )


def main() -> int:
    try:
        bundle = load_bundle()
        validate_matched_failures(*bundle)
        families, scenarios, variants, episodes = statistics(bundle[5])
    except MatchedFailureValidationError as exc:
        print(f"Matched failure validation failed:\n{exc}", file=sys.stderr)
        return 1
    print("Matched failure validation passed.")
    print(f"Families: {families}")
    print(f"Scenarios: {scenarios}")
    print(f"Agent Error: {variants['AE']}")
    print(f"Transient Failure: {variants['TF']}")
    print(f"Persistent Drift: {variants['PD']}")
    print(f"Episodes: {episodes}")
    print("Shared first-failure signatures: valid")
    print("Ground-truth leakage: none")
    print("Held-out transfer separation: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
