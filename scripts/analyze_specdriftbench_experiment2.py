#!/usr/bin/env python3
"""Offline, deterministic Experiment 2 analysis for the frozen V2 held-out run.

This module deliberately has no provider, cache, credential, or network imports.
The statistical artifacts are generated with the fixed project interpreter.  The
optional figure renderer uses Pillow and ReportLab only when ``--render-only``
is requested, so the analysis and its tests remain dependency-free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ID = "specdriftbench-heldout432-v2-20260719-3f4fd32-01"
ATTEMPT_ROOT = (
    PROJECT_ROOT
    / "results/experiments/phase11/specdriftbench_component_heldout432/attempts"
    / ATTEMPT_ID
)
OUTPUT_ROOT = ATTEMPT_ROOT / "analysis/experiment2"
LEDGER_PATH = PROJECT_ROOT / "results/experiments/phase10/cost/ledger.json"
ORACLE_AUDIT_PATH = PROJECT_ROOT / "results/attribution/attribution_conformance_v1.json"

CLASSES = ("AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT")
PREDICTION_COLUMNS = CLASSES + ("NOT_EVALUABLE",)
VIEWS = ("FIRST_FAILURE", "RETRY_HISTORY", "FULL_EVIDENCE")
PROVIDERS = ("deepseek", "dashscope", "moonshot")
PROVIDER_LABEL = {"deepseek": "DeepSeek", "dashscope": "Qwen", "moonshot": "Kimi"}
VARIANT_CLASS = {
    "AE": "AGENT_ERROR",
    "TF": "TRANSIENT_FAILURE",
    "PD": "PERSISTENT_DRIFT",
}
COMPARISONS = (
    ("RETRY_HISTORY", "FIRST_FAILURE"),
    ("FULL_EVIDENCE", "FIRST_FAILURE"),
    ("FULL_EVIDENCE", "RETRY_HISTORY"),
)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_718
CONFIDENCE_LEVEL = 0.95

EXPECTED_FINGERPRINTS = {
    "prompt_sha256": "d499cdb339972ad5a9424339519f4c64a9cb9096afcddd46b6309ccd7cc2a47e",
    "attribution_schema_sha256": "fb1a66c6d331fc1af98ef6985518f19e08ef9733f2ad07624dbfd962ce44b6f9",
    "evidence_view_sha256": "94376b30b8554539602922f6da656dedcfc7a376650daec64735658af39585ac",
    "tool_registry_sha256": "5be5dacb739ed05cb81b5a005de0faf414312d9bb9ef0664eafe8a1000ed7205",
}
EXPECTED_ANALYSIS_PLAN = "c0f52ae987c2600be71685b14c3d171d2cbd34651bd2db11919b621ca6101d71"
EXPECTED_VIEW_MACRO_F1 = {
    "FIRST_FAILURE": 0.34398034398034394,
    "RETRY_HISTORY": 0.4932285764649593,
    "FULL_EVIDENCE": 0.4854802680565898,
}
EXPECTED_PAIRED_CI = {
    "RETRY_HISTORY - FIRST_FAILURE": [0.020833333333333332, 0.13194444444444445],
    "FULL_EVIDENCE - FIRST_FAILURE": [0.041666666666666664, 0.1527777777777778],
    "FULL_EVIDENCE - RETRY_HISTORY": [-0.04861111111111111, 0.09027777777777778],
}
EXPECTED_SOURCE_HASHES = {
    "records_tree_sha256": "bee6b8c065a1f4612db1ae8da482717fc236043f25c8075e143431cc36712499",
    "manifest_sha256": "895f129bcf38a20ad414203b0b45a2aa3710e2deb2b042dc524bb8d524c61a1b",
    "summary_sha256": "1fbb5b31d4563a74fa70938b16354a71237143718b434bf4ea168bd056592f22",
    "heldout_analysis_v2_sha256": "ebaa34388e86a7dd256443f40f625c37b5f7ee015eebbb4be514f8ca97153ffc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def predicted_class(record: dict[str, Any]) -> str:
    if not record.get("schema_valid"):
        return "NOT_EVALUABLE"
    prediction = record.get("normalized_prediction") or record.get("parsed_result")
    if not isinstance(prediction, dict):
        return "NOT_EVALUABLE"
    value = prediction.get("predicted_class")
    return value if value in CLASSES else "NOT_EVALUABLE"


def actual_class(record: dict[str, Any]) -> str:
    return VARIANT_CLASS[record["variant"]]


def classification_from_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in PREDICTION_COLUMNS} for actual in CLASSES}
    records = 0
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1
        records += 1
    evaluable = sum(matrix[a][p] for a in CLASSES for p in CLASSES)
    correct = sum(matrix[label][label] for label in CLASSES)
    per_class: dict[str, Any] = {}
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in CLASSES if actual != label)
        fn = sum(matrix[label][predicted] for predicted in PREDICTION_COLUMNS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    return {
        "records": records,
        "evaluable_records": evaluable,
        "coverage": evaluable / records if records else 0.0,
        "accuracy_all_records": correct / records if records else 0.0,
        "accuracy_evaluable_subset": correct / evaluable if evaluable else 0.0,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(CLASSES),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(CLASSES),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def classification(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return classification_from_pairs((actual_class(row), predicted_class(row)) for row in records)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def confidence_interval(values: Sequence[float]) -> list[float]:
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return [percentile(values, alpha), percentile(values, 1.0 - alpha)]


def paired_rows(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        groups[row["paired_group_id"]][row["evidence_view"]] = row
    if len(groups) != 144 or any(set(group) != set(VIEWS) for group in groups.values()):
        raise ValueError("Expected 144 complete three-view paired groups")
    return dict(groups)


def paired_comparison(
    groups: dict[str, dict[str, dict[str, Any]]], view_a: str, view_b: str
) -> dict[str, Any]:
    transition = Counter()
    joint_a_correct = joint_b_correct = joint_evaluable = 0
    coverage_a = coverage_b = 0
    rows = []
    for pair_id in sorted(groups):
        a = groups[pair_id][view_a]
        b = groups[pair_id][view_b]
        pred_a, pred_b = predicted_class(a), predicted_class(b)
        truth = actual_class(a)
        correct_a, correct_b = pred_a == truth, pred_b == truth
        eval_a, eval_b = pred_a != "NOT_EVALUABLE", pred_b != "NOT_EVALUABLE"
        coverage_a += eval_a
        coverage_b += eval_b
        if correct_a and not correct_b:
            transition["a_correct_b_incorrect"] += 1
        elif correct_b and not correct_a:
            transition["a_incorrect_b_correct"] += 1
        elif correct_a and correct_b:
            transition["both_correct"] += 1
        else:
            transition["both_incorrect"] += 1
        if eval_a and eval_b:
            joint_evaluable += 1
            joint_a_correct += correct_a
            joint_b_correct += correct_b
        rows.append(
            {
                "provider": a["provider"],
                "family": a["family"],
                "variant": a["variant"],
                "paired_group_id": pair_id,
                "view_a": view_a,
                "view_b": view_b,
                "actual_class": truth,
                "prediction_a": pred_a,
                "prediction_b": pred_b,
                "correct_a": int(correct_a),
                "correct_b": int(correct_b),
                "evaluable_a": int(eval_a),
                "evaluable_b": int(eval_b),
            }
        )
    total = len(groups)
    result = {
        "view_a": view_a,
        "view_b": view_b,
        "paired_groups": total,
        "accuracy_delta_all_records": (transition["a_correct_b_incorrect"] - transition["a_incorrect_b_correct"]) / total,
        "coverage_delta": (coverage_a - coverage_b) / total,
        "jointly_evaluable": joint_evaluable,
        "jointly_evaluable_rate": joint_evaluable / total,
        "accuracy_delta_jointly_evaluable": (
            (joint_a_correct - joint_b_correct) / joint_evaluable if joint_evaluable else 0.0
        ),
        "transitions": dict(sorted(transition.items())),
        "rows": rows,
    }
    for key in ("a_correct_b_incorrect", "a_incorrect_b_correct", "both_correct", "both_incorrect"):
        result["transitions"].setdefault(key, 0)
    return result


def bootstrap_statistics(
    records: Sequence[dict[str, Any]], repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[row["family"]].append(row)
    families = sorted(by_family)
    if len(families) != 16 or any(len(by_family[family]) != 27 for family in families):
        raise ValueError("Family-cluster bootstrap requires 16 intact 27-record clusters")
    rng = random.Random(seed)
    view_samples = {view: {"accuracy": [], "macro_f1": []} for view in VIEWS}
    comparison_samples = {f"{a} - {b}": [] for a, b in COMPARISONS}
    for _ in range(repetitions):
        sampled = [rng.choice(families) for _ in families]
        rows = [row for family in sampled for row in by_family[family]]
        metrics = {
            view: classification(row for row in rows if row["evidence_view"] == view)
            for view in VIEWS
        }
        for view in VIEWS:
            view_samples[view]["accuracy"].append(metrics[view]["accuracy_all_records"])
            view_samples[view]["macro_f1"].append(metrics[view]["macro_f1"])
        for view_a, view_b in COMPARISONS:
            comparison_samples[f"{view_a} - {view_b}"].append(
                metrics[view_a]["accuracy_all_records"] - metrics[view_b]["accuracy_all_records"]
            )
    return {
        "settings": {
            "unit": "family",
            "families_per_replicate": 16,
            "repetitions": repetitions,
            "seed": seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "interval": "percentile",
        },
        "view_intervals": {
            view: {
                "accuracy_ci95": confidence_interval(view_samples[view]["accuracy"]),
                "macro_f1_ci95": confidence_interval(view_samples[view]["macro_f1"]),
            }
            for view in VIEWS
        },
        "paired_accuracy_delta_intervals": {
            name: confidence_interval(values) for name, values in comparison_samples.items()
        },
    }


def validate_source_identity(manifest: dict[str, Any], frozen: dict[str, Any]) -> None:
    if manifest.get("attempt_id") != ATTEMPT_ID or frozen.get("attempt_id") != ATTEMPT_ID:
        raise ValueError("Only the formal V2 held-out Attempt is admissible")
    if manifest.get("execution_status") != "REAL_PROVIDER" or manifest.get("infrastructure_gate_version") != 2:
        raise ValueError("Source is not the formal V2 real-provider protocol")
    if (
        manifest.get("counts", {}).get("records") != 432
        or frozen.get("status") != "COMPLETE"
        or frozen.get("records_completed") != 432
        or frozen.get("records_planned") != 432
    ):
        raise ValueError("Formal V2 source is incomplete")
    if manifest.get("analysis_plan_sha256") != EXPECTED_ANALYSIS_PLAN:
        raise ValueError("Frozen analysis-plan fingerprint changed")
    if manifest.get("frozen_fingerprints") != {
        **EXPECTED_FINGERPRINTS,
        "pricing_sha256": manifest["frozen_fingerprints"].get("pricing_sha256"),
    }:
        raise ValueError("Frozen asset fingerprint changed")


def load_and_validate(attempt_root: Path = ATTEMPT_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path = attempt_root / "results/manifest.json"
    summary_path = attempt_root / "results/summary.json"
    analysis_path = attempt_root / "analysis/heldout_analysis_v2.json"
    records_path = attempt_root / "results/records"
    manifest, summary, frozen = map(read_json, (manifest_path, summary_path, analysis_path))
    validate_source_identity(manifest, frozen)
    source_hashes = {
        "records_tree_sha256": tree_hash(records_path),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "heldout_analysis_v2_sha256": sha256_file(analysis_path),
    }
    if source_hashes != EXPECTED_SOURCE_HASHES:
        raise ValueError(f"Formal V2 source bytes changed: {source_hashes}")
    records = [read_json(path) for path in sorted(records_path.glob("*.json"))]
    if len(records) != 432 or any(row.get("attempt_id") != ATTEMPT_ID for row in records):
        raise ValueError("Expected exactly 432 formal V2 records")
    if set(row["family"] for row in records) & set(manifest["development_families"]):
        raise ValueError("Development family leaked into held-out analysis")
    if Counter(row["provider"] for row in records) != Counter({provider: 144 for provider in PROVIDERS}):
        raise ValueError("Provider balance changed")
    if Counter(row["evidence_view"] for row in records) != Counter({view: 144 for view in VIEWS}):
        raise ValueError("Evidence-view balance changed")
    if Counter(row["variant"] for row in records) != Counter({variant: 144 for variant in VARIANT_CLASS}):
        raise ValueError("Variant balance changed")
    if any(row.get("repetition") != 1 for row in records):
        raise ValueError("Repetition is not frozen at one")
    if sum(row.get("content_length") is not None for row in records) != 432:
        raise ValueError("Response availability is not 432/432")
    if sum(predicted_class(row) != "NOT_EVALUABLE" for row in records) != 427:
        raise ValueError("Evaluable/non-evaluable split changed")
    groups = paired_rows(records)
    tuple_keys = {
        (
            group["FIRST_FAILURE"]["provider"],
            group["FIRST_FAILURE"]["family"],
            group["FIRST_FAILURE"]["variant"],
        )
        for group in groups.values()
    }
    if len(tuple_keys) != 144:
        raise ValueError("Provider + family + variant pairing key is not unique")
    return records, manifest, {"summary": summary, "frozen": frozen, "source_hashes": source_hashes}


def provider_view_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        provider: {
            view: classification(
                row for row in records if row["provider"] == provider and row["evidence_view"] == view
            )
            for view in VIEWS
        }
        for provider in PROVIDERS
    }


def provider_paired_deltas(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for provider in PROVIDERS:
        subset = {
            pair_id: group
            for pair_id, group in groups.items()
            if group["FIRST_FAILURE"]["provider"] == provider
        }
        result[provider] = {
            f"{a} - {b}": paired_comparison(subset, a, b)["accuracy_delta_all_records"]
            for a, b in COMPARISONS
        }
    return result


def label_view_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in CLASSES:
        result[label] = {}
        for view in VIEWS:
            rows = [row for row in records if actual_class(row) == label and row["evidence_view"] == view]
            predictions = [predicted_class(row) for row in rows]
            correct = predictions.count(label)
            non_evaluable = predictions.count("NOT_EVALUABLE")
            result[label][view] = {
                "records": len(rows),
                "correct": correct,
                "incorrect": len(rows) - correct - non_evaluable,
                "non_evaluable": non_evaluable,
                "recall_all_records": correct / len(rows),
                "coverage": (len(rows) - non_evaluable) / len(rows),
            }
        result[label]["recall_deltas"] = {
            f"{a} - {b}": result[label][a]["recall_all_records"] - result[label][b]["recall_all_records"]
            for a, b in COMPARISONS
        }
    return result


def baselines(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    always_pd = classification_from_pairs(
        (actual_class(row), "PERSISTENT_DRIFT") for row in records
    )
    oracle_source = read_json(ORACLE_AUDIT_PATH) if ORACLE_AUDIT_PATH.exists() else {}
    phase7_count = (
        oracle_source.get("summary", {}).get("scenarios", 0)
        if isinstance(oracle_source, dict)
        else 0
    )
    return {
        "uniform_random": {
            "label": "ANALYTICAL_EXPECTED_BASELINE",
            "method": "closed-form expectation for balanced three-class labels",
            "accuracy": 1 / 3,
            "macro_precision": 1 / 3,
            "macro_recall": 1 / 3,
            "macro_f1": 1 / 3,
            "samples_drawn": 0,
        },
        "always_persistent_drift": {
            "label": "DETERMINISTIC_BASELINE",
            **always_pd,
        },
        "symbolic_oracle": {
            "value": None,
            "display": "N/A",
            "reason": "protocol alignment not proven",
            "label": "NON_COMPARABLE_SYMBOLIC_ORACLE_UPPER_BOUND",
            "audit": {
                "phase7_source_present": ORACLE_AUDIT_PATH.exists(),
                "phase7_result_count": phase7_count,
                "required_heldout_scenarios": 48,
                "identical_family_set_proven": False,
                "identical_evidence_mapping_proven": False,
            },
        },
    }


def prediction_phenomena(records: Sequence[dict[str, Any]], views: dict[str, Any]) -> dict[str, Any]:
    pd_counts: dict[str, Any] = {}
    for view in VIEWS:
        rows = [row for row in records if row["evidence_view"] == view]
        predicted_pd = [row for row in rows if predicted_class(row) == "PERSISTENT_DRIFT"]
        pd_counts[view] = {
            "count": len(predicted_pd),
            "proportion": len(predicted_pd) / len(rows),
            "from_actual_class": dict(sorted(Counter(actual_class(row) for row in predicted_pd).items())),
        }
    return {
        "accuracy_macro_f1_divergence": {
            view: {
                "accuracy": views[view]["accuracy_all_records"],
                "macro_f1": views[view]["macro_f1"],
                "gap_accuracy_minus_macro_f1": views[view]["accuracy_all_records"] - views[view]["macro_f1"],
                "per_class": views[view]["per_class"],
            }
            for view in VIEWS
        },
        "persistent_drift_prediction_concentration": pd_counts,
        "interpretation": (
            "Persistent-drift predictions remain concentrated across views. Retry evidence primarily restores "
            "transient-failure recall, while full evidence raises agent-error recall but gives back part of the "
            "transient-failure gain; this explains why accuracy and Macro-F1 order the two richest views differently."
        ),
    }


def build_report(
    records: Sequence[dict[str, Any]], manifest: dict[str, Any], source: dict[str, Any], repetitions: int = BOOTSTRAP_REPETITIONS
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = paired_rows(records)
    views = {view: classification(row for row in records if row["evidence_view"] == view) for view in VIEWS}
    for view, expected in EXPECTED_VIEW_MACRO_F1.items():
        if not math.isclose(views[view]["macro_f1"], expected, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"{view} Macro-F1 does not reproduce frozen analysis")
    bootstrap = bootstrap_statistics(records, repetitions=repetitions)
    comparisons: dict[str, Any] = {}
    transition_rows: list[dict[str, Any]] = []
    for view_a, view_b in COMPARISONS:
        name = f"{view_a} - {view_b}"
        comparison = paired_comparison(groups, view_a, view_b)
        transition_rows.extend(comparison.pop("rows"))
        comparison["ci95_family_clustered_bootstrap"] = bootstrap["paired_accuracy_delta_intervals"][name]
        comparison["ci_excludes_zero"] = comparison["ci95_family_clustered_bootstrap"][0] > 0 or comparison["ci95_family_clustered_bootstrap"][1] < 0
        comparisons[name] = comparison
        if repetitions == BOOTSTRAP_REPETITIONS:
            if any(
                not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-15)
                for observed, expected in zip(
                    comparison["ci95_family_clustered_bootstrap"], EXPECTED_PAIRED_CI[name]
                )
            ):
                raise ValueError(f"{name} CI does not reproduce frozen analysis")
            # Preserve the exact decimal representation in the frozen V2 artifact.
            comparison["ci95_family_clustered_bootstrap"] = EXPECTED_PAIRED_CI[name]
    by_provider_view = provider_view_metrics(records)
    labels = label_view_metrics(records)
    error_counts = Counter(row.get("error_class", "NONE") for row in records)
    infrastructure_classes = {
        "AUTHENTICATION",
        "AUTHORIZATION",
        "BALANCE_EXHAUSTED",
        "MODEL_NOT_FOUND",
        "NETWORK",
        "PROVIDER_CONFIGURATION",
        "RATE_LIMIT",
        "SERVER",
        "TIMEOUT",
        "UNKNOWN_PROVIDER_ERROR",
    }
    quality = {
        "schema_valid": sum(bool(row.get("schema_valid")) for row in records),
        "schema_valid_rate": sum(bool(row.get("schema_valid")) for row in records) / len(records),
        "not_evaluable": sum(predicted_class(row) == "NOT_EVALUABLE" for row in records),
        "invalid_structured_output": error_counts["INVALID_STRUCTURED_OUTPUT"],
        "output_truncated": error_counts["OUTPUT_TRUNCATED"],
        "infrastructure_errors": sum(error_counts[name] for name in infrastructure_classes),
        "error_class_counts": dict(sorted((name, count) for name, count in error_counts.items() if name != "NONE")),
        "leakage_failures": sum(not row.get("leakage", {}).get("passed", False) for row in records),
        "fallbacks": sum(bool(row.get("fallback_used")) for row in records),
    }
    report = {
        "schema_version": "specdriftbench-experiment2-analysis-v1",
        "status": "EXPERIMENT_2_COMPLETE",
        "attempt_id": ATTEMPT_ID,
        "analysis_scope": "OFFLINE_EVIDENCE_ABLATION_AND_BASELINE_COMPARISON",
        "records": len(records),
        "families": len(set(row["family"] for row in records)),
        "source_integrity": {
            **source["source_hashes"],
            "analysis_plan_sha256": manifest["analysis_plan_sha256"],
            "frozen_fingerprints": manifest["frozen_fingerprints"],
            "ledger_sha256_at_analysis_start": sha256_file(LEDGER_PATH),
            "v1_classification": "INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER",
            "v1_used": False,
        },
        "denominator_policy": {
            "accuracy_all_records": "correct / all records; non-evaluable records are not correct",
            "accuracy_evaluable_subset": "correct / evaluable records",
            "coverage": "evaluable records / all records",
            "class_recall": "correct records for class / all actual records for class",
            "paired_delta": "(correct under view A - correct under view B) / all paired groups",
        },
        "by_evidence_view": views,
        "bootstrap": bootstrap,
        "paired_comparisons": comparisons,
        "by_provider_and_evidence_view": by_provider_view,
        "provider_paired_accuracy_deltas_descriptive": provider_paired_deltas(groups),
        "by_label_and_evidence_view": labels,
        "baselines": baselines(records),
        "phenomena": prediction_phenomena(records, views),
        "quality": quality,
        "safety": {
            "api_calls": 0,
            "provider_instances": 0,
            "network_calls": 0,
            "cache_reads": 0,
            "credentials_read": 0,
            "ground_truth_exposed_to_provider": False,
        },
        "claims": {
            "retry_vs_first": "SUPPORTED: positive paired accuracy gain; family-clustered 95% CI excludes zero.",
            "full_vs_first": "SUPPORTED: positive paired accuracy gain; family-clustered 95% CI excludes zero.",
            "full_vs_retry": "NOT SUPPORTED as an additional accuracy gain: 95% CI crosses zero.",
            "diminishing_returns": "Retry history captures most of the observed accuracy gain; full evidence adds no clear further gain over retry history.",
            "prompt_tuning": "PROHIBITED: formal held-out outcomes are reported without system changes.",
        },
    }
    return report, transition_rows


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    views = report["by_evidence_view"]
    lines = [
        "# Experiment 2: Evidence Ablation and Baseline Comparison",
        "",
        f"- Attempt: `{report['attempt_id']}`",
        "- Scope: frozen V2 held-out records only; fully offline",
        f"- Records: {report['records']} across {report['families']} held-out families",
        "- Bootstrap: family-clustered, 10,000 repetitions, seed 20260718, percentile 95% CI",
        "",
        "## Evidence-view performance",
        "",
        "| Evidence view | Accuracy (all) | Accuracy (evaluable) | Coverage | Macro-P | Macro-R | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view in VIEWS:
        metric = views[view]
        lines.append(
            f"| {view} | {format_number(metric['accuracy_all_records'])} | "
            f"{format_number(metric['accuracy_evaluable_subset'])} | {format_number(metric['coverage'])} | "
            f"{format_number(metric['macro_precision'])} | {format_number(metric['macro_recall'])} | "
            f"{format_number(metric['macro_f1'])} |"
        )
    lines += ["", "## Paired evidence gains", "", "| Comparison | All-record delta | 95% CI | Joint-evaluable delta | Coverage delta | A+/B- | A-/B+ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in report["paired_comparisons"].items():
        ci = metric["ci95_family_clustered_bootstrap"]
        t = metric["transitions"]
        lines.append(
            f"| {name} | {format_number(metric['accuracy_delta_all_records'])} | "
            f"[{format_number(ci[0])}, {format_number(ci[1])}] | "
            f"{format_number(metric['accuracy_delta_jointly_evaluable'])} | {format_number(metric['coverage_delta'])} | "
            f"{t['a_correct_b_incorrect']} | {t['a_incorrect_b_correct']} |"
        )
    lines += [
        "",
        "The Retry–First and Full–First intervals exclude zero. The Full–Retry interval crosses zero, so the formal data do not support a clear additional accuracy gain from full evidence over retry history. Retry history captures most of the observed gain.",
        "",
        "## Baselines",
        "",
        "| Baseline | Accuracy | Macro-P | Macro-R | Macro-F1 | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    uniform = report["baselines"]["uniform_random"]
    always = report["baselines"]["always_persistent_drift"]
    oracle = report["baselines"]["symbolic_oracle"]
    lines += [
        f"| Uniform random (analytical expectation) | {format_number(uniform['accuracy'])} | {format_number(uniform['macro_precision'])} | {format_number(uniform['macro_recall'])} | {format_number(uniform['macro_f1'])} | {uniform['label']} |",
        f"| Always-PD | {format_number(always['accuracy_all_records'])} | {format_number(always['macro_precision'])} | {format_number(always['macro_recall'])} | {format_number(always['macro_f1'])} | {always['label']} |",
        f"| Symbolic Oracle | N/A | N/A | N/A | N/A | {oracle['label']}: {oracle['reason']} |",
        "",
        "FIRST_FAILURE Macro-F1 is only slightly above the analytical random expectation (0.3440 vs 0.3333). The richer views improve class balance, but persistent-drift overprediction remains the dominant error pattern.",
        "",
        "## Class-level mechanism",
        "",
        "| Actual class | View | Correct | Incorrect | Non-evaluable | Recall | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in CLASSES:
        for view in VIEWS:
            metric = report["by_label_and_evidence_view"][label][view]
            lines.append(f"| {label} | {view} | {metric['correct']} | {metric['incorrect']} | {metric['non_evaluable']} | {format_number(metric['recall_all_records'])} | {format_number(metric['coverage'])} |")
    lines += [
        "",
        report["phenomena"]["interpretation"],
        "",
        "## Validity and use",
        "",
        "This is a confirmatory offline analysis of the frozen formal V2 held-out Attempt. V1 is excluded as `INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER`. No API, provider, cache, credential, prompt, schema, evidence, benchmark, or model path was used or changed. The results are suitable for the paper as evidence-ablation and deterministic/analytical baseline comparisons, subject to the stated denominator and oracle non-comparability notes.",
        "",
    ]
    return "\n".join(lines)


def render_paper_tables(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment 2 paper tables",
        "",
        "## Table E2.1 — Evidence ablation",
        "",
        "| Evidence | Accuracy | Macro-F1 | Accuracy 95% CI | Macro-F1 95% CI | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view in VIEWS:
        metric = report["by_evidence_view"][view]
        interval = report["bootstrap"]["view_intervals"][view]
        lines.append(
            f"| {view} | {format_number(metric['accuracy_all_records'])} | {format_number(metric['macro_f1'])} | "
            f"[{format_number(interval['accuracy_ci95'][0])}, {format_number(interval['accuracy_ci95'][1])}] | "
            f"[{format_number(interval['macro_f1_ci95'][0])}, {format_number(interval['macro_f1_ci95'][1])}] | {format_number(metric['coverage'])} |"
        )
    lines += [
        "",
        "## Table E2.2 — Baseline comparison",
        "",
        "| Method/View | Accuracy | Macro-P | Macro-R | Macro-F1 | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    uniform = report["baselines"]["uniform_random"]
    always = report["baselines"]["always_persistent_drift"]
    lines.append(f"| Uniform Random Expected | {format_number(uniform['accuracy'])} | {format_number(uniform['macro_precision'])} | {format_number(uniform['macro_recall'])} | {format_number(uniform['macro_f1'])} | 1.0000 |")
    lines.append(f"| Always-PD | {format_number(always['accuracy_all_records'])} | {format_number(always['macro_precision'])} | {format_number(always['macro_recall'])} | {format_number(always['macro_f1'])} | {format_number(always['coverage'])} |")
    for view in VIEWS:
        metric = report["by_evidence_view"][view]
        lines.append(f"| {view} | {format_number(metric['accuracy_all_records'])} | {format_number(metric['macro_precision'])} | {format_number(metric['macro_recall'])} | {format_number(metric['macro_f1'])} | {format_number(metric['coverage'])} |")
    lines.append("| Symbolic Oracle Upper Bound | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "## Table E2.3 — Provider × evidence-view performance", "", "| Provider | View | Accuracy | Coverage | Macro-P | Macro-R | Macro-F1 | AE Recall | TF Recall | PD Recall |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for provider in PROVIDERS:
        pv = report["by_provider_and_evidence_view"][provider]
        for view in VIEWS:
            metric = pv[view]
            lines.append(f"| {PROVIDER_LABEL[provider]} | {view} | {format_number(metric['accuracy_all_records'])} | {format_number(metric['coverage'])} | {format_number(metric['macro_precision'])} | {format_number(metric['macro_recall'])} | {format_number(metric['macro_f1'])} | {format_number(metric['per_class']['AGENT_ERROR']['recall'])} | {format_number(metric['per_class']['TRANSIENT_FAILURE']['recall'])} | {format_number(metric['per_class']['PERSISTENT_DRIFT']['recall'])} |")
    lines += ["", "### Provider-specific comparison with simple baselines", "", "| Provider | Method/View | Accuracy | Macro-F1 | Coverage |", "|---|---|---:|---:|---:|"]
    for provider in PROVIDERS:
        label = PROVIDER_LABEL[provider]
        lines.append(f"| {label} | Uniform Random Expected | 0.3333 | 0.3333 | 1.0000 |")
        lines.append(f"| {label} | Always-PD | 0.3333 | 0.1667 | 1.0000 |")
        for view in VIEWS:
            metric = report["by_provider_and_evidence_view"][provider][view]
            lines.append(f"| {label} | {view} | {format_number(metric['accuracy_all_records'])} | {format_number(metric['macro_f1'])} | {format_number(metric['coverage'])} |")
    lines += ["", "## Table E2.4 — Paired accuracy differences", "", "| Comparison | Delta | Family-clustered 95% CI |", "|---|---:|---:|"]
    for name, metric in report["paired_comparisons"].items():
        ci = metric["ci95_family_clustered_bootstrap"]
        lines.append(f"| {name} | {format_number(metric['accuracy_delta_all_records'])} | [{format_number(ci[0])}, {format_number(ci[1])}] |")
    lines += ["", "Notes: all-record accuracy retains non-evaluable records in the denominator. Bootstrap unit is family (16 clusters), 10,000 repetitions, seed 20260718. Symbolic Oracle is N/A because strict protocol alignment is not proven.", ""]
    return "\n".join(lines)


def render_baselines_markdown(report: dict[str, Any]) -> str:
    baseline = report["baselines"]
    uniform = baseline["uniform_random"]
    always = baseline["always_persistent_drift"]
    oracle = baseline["symbolic_oracle"]
    return "\n".join(
        [
            "# Experiment 2 baselines",
            "",
            "| Baseline | Accuracy | Macro-P | Macro-R | Macro-F1 | Coverage | Classification |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| Uniform Random Expected | {format_number(uniform['accuracy'])} | {format_number(uniform['macro_precision'])} | {format_number(uniform['macro_recall'])} | {format_number(uniform['macro_f1'])} | 1.0000 | {uniform['label']} |",
            f"| Always-PD | {format_number(always['accuracy_all_records'])} | {format_number(always['macro_precision'])} | {format_number(always['macro_recall'])} | {format_number(always['macro_f1'])} | {format_number(always['coverage'])} | {always['label']} |",
            f"| Symbolic Oracle Upper Bound | N/A | N/A | N/A | N/A | N/A | {oracle['label']} |",
            "",
            "Uniform Random is a closed-form expectation and is not an API run. Always-PD is recomputed programmatically across all 432 records. Symbolic Oracle is N/A because protocol alignment is not proven: the Phase 7 result uses benchmark-specific deterministic rules, is not a fair LLM baseline, cannot participate in model ranking, and is only eligible as a solvability check after strict alignment.",
            "",
        ]
    )


def write_csv_outputs(output_root: Path, report: dict[str, Any], transitions: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for view in VIEWS:
        metric = report["by_evidence_view"][view]
        rows.append({"scope": "overall", "provider": "ALL", "evidence_view": view, "records": metric["records"], "accuracy_all_records": metric["accuracy_all_records"], "accuracy_evaluable_subset": metric["accuracy_evaluable_subset"], "coverage": metric["coverage"], "macro_precision": metric["macro_precision"], "macro_recall": metric["macro_recall"], "macro_f1": metric["macro_f1"]})
    for provider in PROVIDERS:
        for view in VIEWS:
            metric = report["by_provider_and_evidence_view"][provider][view]
            rows.append({"scope": "provider", "provider": provider, "evidence_view": view, "records": metric["records"], "accuracy_all_records": metric["accuracy_all_records"], "accuracy_evaluable_subset": metric["accuracy_evaluable_subset"], "coverage": metric["coverage"], "macro_precision": metric["macro_precision"], "macro_recall": metric["macro_recall"], "macro_f1": metric["macro_f1"]})
    fieldnames = list(rows[0])
    with (output_root / "experiment2_provider_view_metrics_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    transition_fields = list(transitions[0])
    with (output_root / "experiment2_paired_transitions_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=transition_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(transitions)
    label_rows: list[dict[str, Any]] = []
    for label in CLASSES:
        for view in VIEWS:
            metric = report["by_label_and_evidence_view"][label][view]
            label_rows.append({"actual_class": label, "evidence_view": view, **metric})
    with (output_root / "experiment2_label_view_metrics_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(label_rows)


def run_analysis(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    records, manifest, source = load_and_validate()
    ledger_before = sha256_file(LEDGER_PATH)
    ledger_values = read_json(LEDGER_PATH)
    report, transitions = build_report(records, manifest, source)
    report["source_integrity"]["ledger_at_analysis_start"] = {
        key: ledger_values[key]
        for key in (
            "api_attempts",
            "provider_reported_input_tokens",
            "provider_reported_output_tokens",
            "spent_cny",
            "reserved_cny",
        )
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment2_evidence_ablation_v1.json", report)
    (output_root / "experiment2_evidence_ablation_v1.md").write_text(render_markdown(report), encoding="utf-8")
    write_json(
        output_root / "experiment2_baselines_v1.json",
        {
            "schema_version": "specdriftbench-experiment2-baselines-v1",
            "attempt_id": ATTEMPT_ID,
            "baselines": report["baselines"],
        },
    )
    (output_root / "experiment2_baselines_v1.md").write_text(render_baselines_markdown(report), encoding="utf-8")
    (output_root / "experiment2_paper_tables_v1.md").write_text(render_paper_tables(report), encoding="utf-8")
    write_csv_outputs(output_root, report, transitions)
    if sha256_file(LEDGER_PATH) != ledger_before:
        raise RuntimeError("Ledger changed during offline analysis")
    return report


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    paths = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


class PaperCanvas:
    def __init__(self, path: Path, width: int = 2100, height: int = 1350, pdf: bool = False):
        self.path, self.width, self.height, self.pdf = path, width, height, pdf
        if pdf:
            from reportlab.pdfgen import canvas
            self.surface = canvas.Canvas(str(path), pagesize=(width, height), invariant=1, pageCompression=1)
            self.surface.setTitle("SpecDriftBench Experiment 2")
            self.surface.setCreator("DriftGuard deterministic renderer")
        else:
            from PIL import Image, ImageDraw
            self.image = Image.new("RGB", (width, height), "#ffffff")
            self.surface = ImageDraw.Draw(self.image)

    def line(self, points: Sequence[tuple[float, float]], fill: str = "#555555", width: int = 3) -> None:
        if self.pdf:
            self.surface.setStrokeColor(fill); self.surface.setLineWidth(width)
            path = self.surface.beginPath(); path.moveTo(points[0][0], self.height - points[0][1])
            for x, y in points[1:]: path.lineTo(x, self.height - y)
            self.surface.drawPath(path)
        else:
            self.surface.line(points, fill=fill, width=width)

    def rect(self, box: tuple[float, float, float, float], fill: str, outline: str = "#333333", width: int = 2, pattern: int = 0) -> None:
        x0, y0, x1, y1 = box
        if self.pdf:
            self.surface.setFillColor(fill); self.surface.setStrokeColor(outline); self.surface.setLineWidth(width)
            self.surface.rect(x0, self.height - y1, x1 - x0, y1 - y0, fill=1, stroke=1)
        else:
            self.surface.rectangle(box, fill=fill, outline=outline, width=width)
        if pattern == 1:
            for x in range(int(x0 + 10), int(x1), 18): self.line([(x, y0 + 2), (x, y1 - 2)], "#555555", 2)
        elif pattern == 2:
            height = y1 - y0
            for start in range(int(x0 - height), int(x1), 22):
                xa, xb = max(x0 + 2, start), min(x1 - 2, start + height)
                ya, yb = y1 - (xa - start), y1 - (xb - start)
                self.line([(xa, ya), (xb, yb)], "#555555", 2)

    def text(self, xy: tuple[float, float], value: str, size: int = 30, anchor: str = "la", bold: bool = False, fill: str = "#202020") -> None:
        x, y = xy
        if self.pdf:
            name = "Helvetica-Bold" if bold else "Helvetica"; self.surface.setFont(name, size); self.surface.setFillColor(fill)
            width = self.surface.stringWidth(value, name, size)
            if anchor.startswith("m"): x -= width / 2
            elif anchor.startswith("r"): x -= width
            self.surface.drawString(x, self.height - y - size * 0.8, value)
        else:
            self.surface.text((x, y), value, font=_font(size, bold), fill=fill, anchor=anchor)

    def circle(self, xy: tuple[float, float], radius: float, fill: str, outline: str = "#333333") -> None:
        x, y = xy; box = (x-radius, y-radius, x+radius, y+radius)
        if self.pdf:
            self.surface.setFillColor(fill); self.surface.setStrokeColor(outline); self.surface.circle(x, self.height-y, radius, fill=1, stroke=1)
        else: self.surface.ellipse(box, fill=fill, outline=outline, width=2)

    def save(self) -> None:
        if self.pdf:
            self.surface.showPage(); self.surface.save()
        else:
            self.image.save(self.path, format="PNG", dpi=(300, 300), optimize=False, compress_level=9)


COLORS = ("#9bb8c4", "#d6b48c", "#a8bea0")
SHORT_VIEW = {"FIRST_FAILURE": "First failure", "RETRY_HISTORY": "Retry history", "FULL_EVIDENCE": "Full evidence"}


def chart_frame(canvas: PaperCanvas, title: str, subtitle: str, y_min: float = 0.0, y_max: float = 1.0):
    canvas.text((1050, 58), title, 46, "ma", True)
    canvas.text((1050, 116), subtitle, 28, "ma", False, "#555555")
    left, top, right, bottom = 185, 205, 2020, 1130
    for index in range(6):
        value = y_min + (y_max-y_min)*index/5
        y = bottom - (value-y_min)/(y_max-y_min)*(bottom-top)
        canvas.line([(left, y), (right, y)], "#d8d8d8", 2)
        canvas.text((left-24, y), f"{value:.2f}", 25, "rm", False, "#555555")
    canvas.line([(left, top), (left, bottom), (right, bottom)], "#333333", 3)
    return left, top, right, bottom


def draw_figure1(canvas: PaperCanvas, report: dict[str, Any]) -> None:
    left, top, right, bottom = chart_frame(canvas, "Evidence-view performance", "All-record accuracy and Macro-F1; family-clustered 95% CI")
    metrics = (("Accuracy", "accuracy_all_records", "accuracy_ci95"), ("Macro-F1", "macro_f1", "macro_f1_ci95"))
    centers = (720, 1510); bar_width = 135
    for group_index, (label, key, ci_key) in enumerate(metrics):
        for view_index, view in enumerate(VIEWS):
            value = report["by_evidence_view"][view][key]
            low, high = report["bootstrap"]["view_intervals"][view][ci_key]
            x = centers[group_index] + (view_index-1)*190
            y = bottom-value*(bottom-top)
            canvas.rect((x-bar_width/2, y, x+bar_width/2, bottom), COLORS[view_index], pattern=view_index)
            canvas.text((x, y-48), f"{value:.3f}", 25, "ma", True)
            y_low, y_high = bottom-low*(bottom-top), bottom-high*(bottom-top)
            canvas.line([(x, y_high), (x, y_low)], "#202020", 3); canvas.line([(x-24,y_high),(x+24,y_high)], "#202020",3); canvas.line([(x-24,y_low),(x+24,y_low)], "#202020",3)
        canvas.text((centers[group_index], bottom+58), label, 30, "ma", True)
    for index, view in enumerate(VIEWS):
        x = 540 + index*520; canvas.rect((x, 1230, x+48, 1270), COLORS[index], pattern=index); canvas.text((x+68, 1250), SHORT_VIEW[view], 27, "lm")


def draw_figure2(canvas: PaperCanvas, report: dict[str, Any]) -> None:
    y_min, y_max = -0.15, 0.20
    left, top, right, bottom = chart_frame(canvas, "Paired accuracy gains", "All 144 paired groups; family-clustered percentile 95% CI", y_min, y_max)
    zero_y = bottom - (0-y_min)/(y_max-y_min)*(bottom-top); canvas.line([(left, zero_y),(right,zero_y)], "#333333", 4)
    names = [f"{a} - {b}" for a,b in COMPARISONS]; centers=(510,1100,1690)
    for index, name in enumerate(names):
        metric=report["paired_comparisons"][name]; value=metric["accuracy_delta_all_records"]; low,high=metric["ci95_family_clustered_bootstrap"]
        y=lambda v: bottom-(v-y_min)/(y_max-y_min)*(bottom-top); x=centers[index]
        canvas.line([(x,y(low)),(x,y(high))], "#3f5966", 8); canvas.line([(x-35,y(low)),(x+35,y(low))], "#3f5966",5); canvas.line([(x-35,y(high)),(x+35,y(high))], "#3f5966",5)
        canvas.circle((x,y(value)), 19, COLORS[index]); canvas.text((x,y(high)-55), f"{value:+.3f}", 27, "ma", True)
        a,b=COMPARISONS[index]; canvas.text((x,bottom+48), f"{SHORT_VIEW[a]} -", 25, "ma", True); canvas.text((x,bottom+82), SHORT_VIEW[b], 25, "ma")
    canvas.text((right,zero_y-18), "no difference", 24, "ra", False, "#555555")


def draw_figure3(canvas: PaperCanvas, report: dict[str, Any]) -> None:
    left, top, right, bottom = chart_frame(canvas, "Provider x evidence-view Macro-F1", "Descriptive provider strata; common 0-1 scale")
    centers=(500,1100,1700); width=135
    for provider_index, provider in enumerate(PROVIDERS):
        for view_index, view in enumerate(VIEWS):
            value=report["by_provider_and_evidence_view"][provider][view]["macro_f1"]
            x=centers[provider_index]+(view_index-1)*170; y=bottom-value*(bottom-top)
            canvas.rect((x-width/2,y,x+width/2,bottom), COLORS[view_index], pattern=view_index)
            canvas.text((x,y-42),f"{value:.3f}",23,"ma",True)
            canvas.text((centers[provider_index],bottom+58),PROVIDER_LABEL[provider],30,"ma",True)
    for index, view in enumerate(VIEWS):
        x=540+index*520; canvas.rect((x,1230,x+48,1270),COLORS[index],pattern=index); canvas.text((x+68,1250),SHORT_VIEW[view],27,"lm")


def render_figures(output_root: Path = OUTPUT_ROOT) -> None:
    report = read_json(output_root / "experiment2_evidence_ablation_v1.json")
    figures = (
        ("figure_evidence_performance_v1", draw_figure1),
        ("figure_paired_evidence_gain_v1", draw_figure2),
        ("figure_provider_evidence_v1", draw_figure3),
    )
    for stem, draw in figures:
        for extension, pdf in (("png", False), ("pdf", True)):
            canvas = PaperCanvas(output_root / f"{stem}.{extension}", pdf=pdf)
            draw(canvas, report); canvas.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analysis-only", action="store_true", help="Generate JSON, Markdown, and CSV outputs")
    mode.add_argument("--render-only", action="store_true", help="Render figures from the generated JSON")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_only:
        render_figures(args.output_root)
    elif args.analysis_only:
        run_analysis(args.output_root)
    else:
        run_analysis(args.output_root)
        render_figures(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
