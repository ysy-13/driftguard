from __future__ import annotations

from collections import defaultdict
from typing import Any


LABELS = ("AE", "TF", "PD")


def confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}
    for row in rows:
        expected, predicted = row.get("_expected"), row.get("predicted_class")
        if expected in LABELS and predicted in LABELS:
            matrix[expected][predicted] += 1
    return matrix


def attribution_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion_matrix(rows)
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    resolved = [row for row in rows if row.get("predicted_class") in LABELS]
    return {
        "confusion_matrix": matrix, "per_class": per_class,
        "accuracy": sum(row.get("_expected") == row.get("predicted_class") for row in rows) / len(rows) if rows else 0.0,
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(LABELS),
        "abstention_rate": 1 - len(resolved) / len(rows) if rows else 0.0,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"records": 0}
    average = lambda key: sum(float(row.get(key, 0) or 0) for row in rows) / len(rows)
    successful = [row for row in rows if row["final_task_success"]]
    proposed = [row for row in rows if row["patch_proposed"]]
    accepted = [row for row in rows if row["patch_accepted"]]
    localized = [row for row in rows if row.get("target_tool_correct") is not None]
    regression_rows = [row for row in rows if row.get("regression_pass") is not None]
    minimality_rows = [row for row in rows if row.get("minimality_pass") is not None]
    return {
        "records": len(rows), "task_success_rate": average("final_task_success"),
        "immediate_repair_success_rate": average("immediate_repair"),
        "future_transfer_rate": average("future_transfer"),
        "forbidden_side_effect_rate": sum(row["forbidden_side_effects"] > 0 for row in rows) / len(rows),
        "unsafe_action_rate": average("unsafe_action"),
        "budget_exhaustion_rate": sum(row["termination_reason"] == "BUDGET_EXHAUSTED" for row in rows) / len(rows),
        "patch_proposal_rate": average("patch_proposed"),
        "raw_patch_correctness": sum(bool(row.get("raw_patch_correct")) for row in proposed) / len(proposed) if proposed else 0.0,
        "patch_validation_survival": len(accepted) / len(proposed) if proposed else 0.0,
        "accepted_patch_correctness": sum(bool(row.get("accepted_patch_correct")) for row in accepted) / len(accepted) if accepted else 0.0,
        "false_patch_rate": average("false_patch"),
        "unsafe_patch_rate": average("unsafe_patch"),
        "regression_pass_rate": sum(bool(row["regression_pass"]) for row in regression_rows) / len(regression_rows) if regression_rows else 0.0,
        "minimality_pass_rate": sum(bool(row["minimality_pass"]) for row in minimality_rows) / len(minimality_rows) if minimality_rows else 0.0,
        "target_tool_accuracy": sum(bool(row["target_tool_correct"]) for row in localized) / len(localized) if localized else 0.0,
        "drift_category_accuracy": sum(bool(row["drift_category_correct"]) for row in localized) / len(localized) if localized else 0.0,
        "exact_location_accuracy": sum(bool(row["exact_location_correct"]) for row in localized) / len(localized) if localized else 0.0,
        "average_llm_calls": average("llm_calls"), "average_tool_calls": average("tool_calls"),
        "average_probe_calls": average("probe_calls"), "average_input_tokens": average("input_tokens"),
        "average_output_tokens": average("output_tokens"), "average_latency_ms": average("latency_ms"),
        "successful_task_average_interactions": (
            sum(row["llm_calls"] + row["tool_calls"] for row in successful) / len(successful) if successful else 0.0
        ),
        "attribution": attribution_metrics(rows),
    }


def by_method_and_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["mode"])].append(row)
    return {f"{method}:{mode}": aggregate_metrics(items) for (method, mode), items in sorted(groups.items())}
