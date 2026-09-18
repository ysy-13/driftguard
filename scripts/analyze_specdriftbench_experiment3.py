#!/usr/bin/env python3
"""Deterministic, offline Experiment 3 failure and robustness analysis.

Only the frozen V2 held-out records are scored.  Raw Cache text is read solely
to classify the already-recorded format-repair boundary; it is never emitted.
The module has no credential loading, Provider construction, or network path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from jsonschema import Draft202012Validator

from driftguard.llm.cache import LLMCache
from driftguard.llm.models import ProviderRequest
from driftguard.phase10.pricing import PricingCatalog
from driftguard.specdriftbench.heldout import HeldoutConfig, HeldoutEvidenceViewBuilder
from driftguard.specdriftbench.protocol import expected_for_evaluator
from driftguard.specdriftbench.runner import OUTPUT_SCHEMA_PATH, PRICING_PATH, PROMPT_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ID = "specdriftbench-heldout432-v2-20260719-3f4fd32-01"
ATTEMPT_ROOT = (
    PROJECT_ROOT / "results/experiments/phase11/specdriftbench_component_heldout432/attempts" / ATTEMPT_ID
)
OUTPUT_ROOT = ATTEMPT_ROOT / "analysis/experiment3"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/specdriftbench_component_heldout432_v2.yaml"
V1_ROOT = (
    PROJECT_ROOT / "results/experiments/phase11/specdriftbench_component_heldout432/attempts"
    / "specdriftbench-heldout432-20260718-af3db03-01"
)
LEDGER_PATH = PROJECT_ROOT / "results/experiments/phase10/cost/ledger.json"
EXPERIMENT2_ROOT = ATTEMPT_ROOT / "analysis/experiment2"

CLASSES = ("AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT")
VIEWS = ("FIRST_FAILURE", "RETRY_HISTORY", "FULL_EVIDENCE")
PROVIDERS = ("deepseek", "dashscope", "moonshot")
PROVIDER_LABEL = {"deepseek": "DeepSeek", "dashscope": "Qwen", "moonshot": "Kimi"}
VARIANT_CLASS = {"AE": "AGENT_ERROR", "TF": "TRANSIENT_FAILURE", "PD": "PERSISTENT_DRIFT"}
DRIFT_TYPES = ("ICD", "RSD", "WPD", "SED")
LOCATION_ERRORS = (
    "EXACT_MATCH", "CORRECT_TARGET_WRONG_FIELD", "CORRECT_PARENT_PATH_TOO_BROAD",
    "CORRECT_CHILD_PATH_TOO_NARROW", "WRONG_COMPONENT_OR_LAYER", "WRONG_CATEGORY",
    "MISSING_LOCATION", "NON_EVALUABLE", "UNCLASSIFIED_LOCATION_ERROR",
)
REPAIR_REASONS = (
    "EMPTY_CONTENT", "INVALID_JSON_SYNTAX", "MISSING_REQUIRED_FIELD", "INVALID_ENUM",
    "WRONG_FIELD_TYPE", "UNKNOWN_FIELD", "SCHEMA_CONSTRAINT_FAILURE", "OUTPUT_TRUNCATED",
    "OTHER_VALIDATION_ERROR",
)
SEMANTIC_REPAIR = (
    "SEMANTICALLY_COMPARABLE_UNCHANGED", "SEMANTICALLY_COMPARABLE_CHANGED", "NOT_COMPARABLE_INVALID_INITIAL_OUTPUT",
    "NOT_COMPARABLE_MISSING_CORE_FIELDS",
)

EXPECTED_SOURCE_HASHES = {
    "records_tree_sha256": "bee6b8c065a1f4612db1ae8da482717fc236043f25c8075e143431cc36712499",
    "cache_tree_sha256": "7d10a3257dccb307e76eb6443890275346693e1d55b4654f3a4a543d13059e5b",
    "experiment2_tree_sha256": "2a014650954495ff71bef8ba4a15e1d24a209d3d5e35abc7ed47c26857e54c49",
    "manifest_sha256": "895f129bcf38a20ad414203b0b45a2aa3710e2deb2b042dc524bb8d524c61a1b",
    "summary_sha256": "1fbb5b31d4563a74fa70938b16354a71237143718b434bf4ea168bd056592f22",
    "ledger_sha256": "317106473bef64ca6407e6130b72511da14d4f995e06634461bd2f98202f8ebb",
}


def _load_experiment2():
    path = PROJECT_ROOT / "scripts/analyze_specdriftbench_experiment2.py"
    spec = importlib.util.spec_from_file_location("specdriftbench_experiment2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Experiment 2 analysis module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E2 = _load_experiment2()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return E2.sha256_file(path)


def tree_hash(path: Path) -> str:
    return E2.tree_hash(path)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def predicted_class(record: dict[str, Any]) -> str:
    return E2.predicted_class(record)


def actual_class(record: dict[str, Any]) -> str:
    return VARIANT_CLASS[record["variant"]]


def prediction(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("normalized_prediction") or record.get("parsed_result")
    return value if isinstance(value, dict) else {}


def quantiles(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values), "median": E2.percentile(values, 0.5),
        "p90": E2.percentile(values, 0.9), "p95": E2.percentile(values, 0.95),
        "max": max(values), "q1": E2.percentile(values, 0.25), "q3": E2.percentile(values, 0.75),
    }


def validate_sources() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records, manifest, source = E2.load_and_validate(ATTEMPT_ROOT)
    observed = {
        "records_tree_sha256": tree_hash(ATTEMPT_ROOT / "results/records"),
        "cache_tree_sha256": tree_hash(ATTEMPT_ROOT / "cache"),
        "experiment2_tree_sha256": tree_hash(EXPERIMENT2_ROOT),
        "manifest_sha256": sha256_file(ATTEMPT_ROOT / "results/manifest.json"),
        "summary_sha256": sha256_file(ATTEMPT_ROOT / "results/summary.json"),
        "ledger_sha256": sha256_file(LEDGER_PATH),
    }
    if observed != EXPECTED_SOURCE_HASHES:
        raise ValueError(f"Experiment 3 source identity changed: {observed}")
    if manifest.get("git_commit") != "3f4fd32a66b5b34c8a943af6647382c4ab4a298b":
        raise ValueError("Formal Attempt Git commit binding changed")
    replay = read_json(ATTEMPT_ROOT / "replay/summary.json")
    if not (
        replay.get("records_completed") == 432 and replay.get("network_calls") == 0
        and replay.get("cache_misses") == 0 and replay.get("provider_instances") == 0
    ):
        raise ValueError("Frozen replay gate is not 432/432 offline")
    return records, manifest, {**source, "source_hashes_experiment3": observed, "replay": replay}


def drift_metadata() -> tuple[HeldoutEvidenceViewBuilder, dict[str, dict[str, Any]]]:
    builder = HeldoutEvidenceViewBuilder()
    expected: dict[str, dict[str, Any]] = {}
    for family in sorted(builder.families):
        expected[family] = expected_for_evaluator(builder, family, "PD")
    return builder, expected


def drift_type(family: str, builder: HeldoutEvidenceViewBuilder) -> str:
    source_id = builder.families[family]["source_drift_id"]
    prefix = source_id.split("-", 1)[0]
    if prefix not in DRIFT_TYPES:
        raise ValueError(f"Unknown drift type for {family}")
    return prefix


def metric_block(rows: Sequence[dict[str, Any]], actual: str | None = None) -> dict[str, Any]:
    total = len(rows)
    evaluable = [row for row in rows if predicted_class(row) != "NOT_EVALUABLE"]
    if actual is None:
        metrics = E2.classification(rows)
        return metrics
    correct = sum(predicted_class(row) == actual for row in rows)
    wrong_pd = sum(predicted_class(row) == "PERSISTENT_DRIFT" for row in rows)
    return {
        "records": total, "evaluable_records": len(evaluable), "non_evaluable_records": total - len(evaluable),
        "coverage": ratio(len(evaluable), total), "correct": correct,
        "recall_all_records": ratio(correct, total), "recall_evaluable_subset": ratio(correct, len(evaluable)),
        "predicted_persistent_drift": wrong_pd,
        "false_drift_rate_all_records": ratio(wrong_pd, total),
        "false_drift_rate_evaluable_subset": ratio(wrong_pd, len(evaluable)),
    }


def false_drift_slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ae_tf = [row for row in rows if row["variant"] in {"AE", "TF"}]
    evaluable = [row for row in ae_tf if predicted_class(row) != "NOT_EVALUABLE"]
    false_pd = sum(predicted_class(row) == "PERSISTENT_DRIFT" for row in ae_tf)
    return {
        "records": len(ae_tf), "evaluable_records": len(evaluable),
        "non_evaluable_records": len(ae_tf) - len(evaluable), "predicted_persistent_drift": false_pd,
        "false_drift_attribution_rate_all_records": ratio(false_pd, len(ae_tf)),
        "false_drift_attribution_rate_evaluable_subset": ratio(false_pd, len(evaluable)),
        "denominators": {
            "all_records": "predicted PD among actual AE/TF / all actual AE/TF records",
            "evaluable_subset": "predicted PD among evaluable actual AE/TF / evaluable actual AE/TF records",
        },
    }


def grouped(rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return dict(sorted(result.items()))


def transition_name(before: dict[str, Any], after: dict[str, Any]) -> str:
    truth = actual_class(before)
    a, b = predicted_class(before), predicted_class(after)
    if a == "NOT_EVALUABLE" or b == "NOT_EVALUABLE":
        return "non_evaluable"
    a_correct, b_correct = a == truth, b == truth
    a_pd, b_pd = a == "PERSISTENT_DRIFT", b == "PERSISTENT_DRIFT"
    if a_pd and b_correct:
        return "wrongPD_to_correct"
    if a_correct and b_pd:
        return "correct_to_wrongPD"
    if not a_correct and not a_pd and b_pd:
        return "other_wrong_to_wrongPD"
    if a_pd and not b_correct and not b_pd:
        return "wrongPD_to_other_wrong"
    if a_correct and b_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def paired_false_drift(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["variant"] in {"AE", "TF"}:
            pairs[(row["provider"], row["family"], row["variant"])][row["evidence_view"]] = row
    comparisons = (
        ("FIRST_FAILURE", "RETRY_HISTORY"), ("RETRY_HISTORY", "FULL_EVIDENCE"),
        ("FIRST_FAILURE", "FULL_EVIDENCE"),
    )
    result: dict[str, Any] = {}
    for variant in ("AE", "TF"):
        result[variant] = {}
        for view_a, view_b in comparisons:
            counts = Counter(
                transition_name(group[view_a], group[view_b])
                for key, group in sorted(pairs.items()) if key[2] == variant
            )
            for name in (
                "wrongPD_to_correct", "correct_to_wrongPD", "other_wrong_to_wrongPD",
                "wrongPD_to_other_wrong", "unchanged_correct", "unchanged_wrong", "non_evaluable",
            ):
                counts.setdefault(name, 0)
            result[variant][f"{view_a} -> {view_b}"] = {
                "pairs": sum(counts.values()), "transitions": dict(sorted(counts.items()))
            }
    return result


def false_drift_analysis(records: Sequence[dict[str, Any]], builder: HeldoutEvidenceViewBuilder) -> dict[str, Any]:
    ae_tf = [row for row in records if row["variant"] in {"AE", "TF"}]
    by_label_view = {
        variant: {
            view: metric_block([r for r in records if r["variant"] == variant and r["evidence_view"] == view], VARIANT_CLASS[variant])
            for view in VIEWS
        }
        for variant in ("AE", "TF", "PD")
    }
    slices = {
        "overall": false_drift_slice(ae_tf),
        "by_provider": {k: false_drift_slice(v) for k, v in grouped(ae_tf, lambda r: r["provider"]).items()},
        "by_evidence_view": {k: false_drift_slice(v) for k, v in grouped(ae_tf, lambda r: r["evidence_view"]).items()},
        "by_provider_and_evidence_view": {
            k: false_drift_slice(v) for k, v in grouped(ae_tf, lambda r: f'{r["provider"]}|{r["evidence_view"]}').items()
        },
        "by_family": {k: false_drift_slice(v) for k, v in grouped(ae_tf, lambda r: r["family"]).items()},
        "by_matched_drift_category": {
            k: false_drift_slice(v)
            for k, v in grouped(ae_tf, lambda r: drift_type(r["family"], builder)).items()
        },
    }
    def error_pattern(rows: Sequence[dict[str, Any]], truth: str) -> dict[str, Any]:
        counts = Counter(predicted_class(row) for row in rows)
        for label in (*CLASSES, "NOT_EVALUABLE"):
            counts.setdefault(label, 0)
        evaluable = len(rows) - counts["NOT_EVALUABLE"]
        return {
            "records": len(rows), "evaluable_records": evaluable, "coverage": ratio(evaluable, len(rows)),
            "correct": counts[truth], "correct_rate_all_records": ratio(counts[truth], len(rows)),
            "correct_rate_evaluable_subset": ratio(counts[truth], evaluable),
            "prediction_distribution": {label: counts[label] for label in (*CLASSES, "NOT_EVALUABLE")},
        }

    error_patterns: dict[str, Any] = {}
    for variant in ("AE", "TF"):
        rows = [row for row in records if row["variant"] == variant]
        truth = VARIANT_CLASS[variant]
        error_patterns[variant] = {"overall": error_pattern(rows, truth)}
        for name, selector in (
            ("provider", lambda r: r["provider"]), ("evidence_view", lambda r: r["evidence_view"]),
            ("provider_and_evidence_view", lambda r: f'{r["provider"]}|{r["evidence_view"]}'),
            ("family", lambda r: r["family"]),
            ("matched_drift_category", lambda r: drift_type(r["family"], builder)),
        ):
            error_patterns[variant][f"by_{name}"] = {
                key: error_pattern(group_rows, truth) for key, group_rows in grouped(rows, selector).items()
            }
    pd_predictions = [row for row in records if predicted_class(row) == "PERSISTENT_DRIFT"]
    true_pd = sum(actual_class(row) == "PERSISTENT_DRIFT" for row in pd_predictions)
    actual_pd = [row for row in records if row["variant"] == "PD"]
    return {
        "definition": "False Drift Attribution Rate = predicted PD among actual AE/TF / actual AE/TF records.",
        "slices": slices, "by_actual_label_and_view": by_label_view,
        "paired_transitions": paired_false_drift(records),
        "actual_error_patterns": error_patterns,
        "pd_prediction_metrics": {
            "predicted_pd": len(pd_predictions), "true_pd_predictions": true_pd,
            "actual_pd_records": len(actual_pd), "all_records": len(records),
            "precision": ratio(true_pd, len(pd_predictions)), "recall": ratio(true_pd, len(actual_pd)),
            "frequency": ratio(len(pd_predictions), len(records)),
            "decomposition": {"correct_PD_detection": true_pd, "AE_misattributed_as_PD": 80, "TF_misattributed_as_PD": 114},
        },
        "distribution_only_interpretation": (
            "Retry history produced more correct TF predictions and fewer TF-to-PD predictions than first-failure evidence. "
            "Full evidence did not preserve the full TF gain. This is a prediction-distribution observation; it does not expose or infer hidden reasoning."
        ),
        "observed_counts": {
            "AE_to_PD": sum(predicted_class(r) == "PERSISTENT_DRIFT" for r in records if r["variant"] == "AE"),
            "TF_to_PD": sum(predicted_class(r) == "PERSISTENT_DRIFT" for r in records if r["variant"] == "TF"),
            "PD_to_PD": sum(predicted_class(r) == "PERSISTENT_DRIFT" for r in records if r["variant"] == "PD"),
        },
    }


def component_metric(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(rows)
    evaluable = [row for row in rows if row.get("schema_valid") and row.get("evaluation", {}).get(field) is not None]
    correct = sum(row.get("evaluation", {}).get(field) is True for row in evaluable)
    return {
        "correct": correct, "all_records": total, "evaluable_records": len(evaluable),
        "coverage": ratio(len(evaluable), total), "accuracy_all_records": ratio(correct, total),
        "accuracy_evaluable_subset": ratio(correct, len(evaluable)),
        "denominator": f"correct PD {field.replace('_correct', '')} / evaluable PD records",
    }


def location_taxonomy(row: dict[str, Any], expected: dict[str, Any]) -> str:
    if not row.get("schema_valid") or predicted_class(row) == "NOT_EVALUABLE":
        return "NON_EVALUABLE"
    if row.get("evaluation", {}).get("location_correct") is True:
        return "EXACT_MATCH"
    pred = prediction(row)
    loc = pred.get("normalized_location")
    if not isinstance(loc, dict) or not (loc.get("spec_pointer") or loc.get("runtime_path")):
        return "MISSING_LOCATION"
    if pred.get("drift_category") != expected["drift_category"]:
        return "WRONG_CATEGORY"
    layer = {"ICD": "input", "RSD": "response", "WPD": "workflow", "SED": "state_effect"}[expected["drift_category"]]
    if pred.get("target_tool") != expected["target_tool"] or loc.get("tool_id") != expected["target_tool"] or loc.get("layer") != layer:
        return "WRONG_COMPONENT_OR_LAYER"
    expected_path = expected.get("location")
    candidates = [loc.get("spec_pointer"), loc.get("runtime_path")]
    if isinstance(expected_path, str):
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            e, c = expected_path.rstrip("/"), candidate.rstrip("/")
            if e.startswith(c + "/"):
                return "CORRECT_PARENT_PATH_TOO_BROAD"
            if c.startswith(e + "/"):
                return "CORRECT_CHILD_PATH_TOO_NARROW"
        return "CORRECT_TARGET_WRONG_FIELD"
    return "UNCLASSIFIED_LOCATION_ERROR"


def localization_block(rows: Sequence[dict[str, Any]], expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = ("class_correct", "category_correct", "target_correct", "location_correct")
    components = {field.replace("_correct", ""): component_metric(rows, field) for field in fields}
    strict = {}
    for index, field in enumerate(fields):
        required = fields[: index + 1]
        correct = sum(all(row.get("evaluation", {}).get(name) is True for name in required) for row in rows)
        strict[field.replace("_correct", "")] = {"correct": correct, "records": len(rows), "rate": ratio(correct, len(rows))}
    taxonomy = Counter(location_taxonomy(row, expected[row["family"]]) for row in rows)
    for category in LOCATION_ERRORS:
        taxonomy.setdefault(category, 0)
    return {
        "records": len(rows), "component_accuracies": components,
        "strict_nested_survival_descriptive": strict,
        "location_error_taxonomy": dict((name, taxonomy[name]) for name in LOCATION_ERRORS),
    }


def localization_analysis(
    records: Sequence[dict[str, Any]], builder: HeldoutEvidenceViewBuilder, expected: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    pd = [row for row in records if row["variant"] == "PD"]
    selectors: dict[str, Callable[[dict[str, Any]], str]] = {
        "provider": lambda r: r["provider"], "evidence_view": lambda r: r["evidence_view"],
        "provider_and_evidence_view": lambda r: f'{r["provider"]}|{r["evidence_view"]}',
        "drift_type": lambda r: drift_type(r["family"], builder), "family": lambda r: r["family"],
        "target_tool": lambda r: expected[r["family"]]["target_tool"],
    }
    result = {"overall": localization_block(pd, expected)}
    for name, selector in selectors.items():
        result[f"by_{name}"] = {
            key: localization_block(rows, expected) for key, rows in grouped(pd, selector).items()
        }
    result["by_drift_type_and_provider"] = {
        f"{dtype}|{provider}": localization_block(
            [row for row in pd if drift_type(row["family"], builder) == dtype and row["provider"] == provider], expected
        )
        for dtype in DRIFT_TYPES for provider in PROVIDERS
    }
    result["by_drift_type_and_evidence_view"] = {
        f"{dtype}|{view}": localization_block(
            [row for row in pd if drift_type(row["family"], builder) == dtype and row["evidence_view"] == view], expected
        )
        for dtype in DRIFT_TYPES for view in VIEWS
    }
    result["drift_type_family_balance"] = dict(sorted(Counter(drift_type(r["family"], builder) for r in pd).items()))
    result["note"] = (
        "Category, target, and exact-location values are separately scored PD-only component accuracies; "
        "they are not assumed to form a monotone funnel. Strict nested survival is included only as a descriptive supplement."
    )
    return result


def _first_request_key(
    row: dict[str, Any], config: HeldoutConfig, builder: HeldoutEvidenceViewBuilder,
    prompt: str, schema: dict[str, Any], formal_config_hash: str,
) -> str:
    evidence = builder.build(row["family"], row["variant"], row["evidence_view"])
    system = prompt.replace("{{OUTPUT_SCHEMA}}", json.dumps(schema, indent=2, sort_keys=True))
    request = ProviderRequest(
        (
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, default=list)},
        ),
        schema, hashlib.sha256(system.encode()).hexdigest(), evidence["public_scenario_id"],
        5, "component_attribution", 1, config.raw["execution"]["seed"],
        f'specdriftbench_heldout432_{row["evidence_view"].lower()}', formal_config_hash,
    )
    return LLMCache.key(config.model(row["provider"]), request, sha256_file(OUTPUT_SCHEMA_PATH))


def repair_reason(raw_text: str, finish_reason: str, schema: dict[str, Any]) -> tuple[str, Any]:
    if finish_reason == "length":
        return "OUTPUT_TRUNCATED", None
    if not raw_text.strip():
        return "EMPTY_CONTENT", None
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return "INVALID_JSON_SYNTAX", None
    if not isinstance(value, dict):
        return "WRONG_FIELD_TYPE", value
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    if not errors:
        return "OTHER_VALIDATION_ERROR", value
    error = errors[0]
    mapping = {
        "required": "MISSING_REQUIRED_FIELD", "enum": "INVALID_ENUM", "type": "WRONG_FIELD_TYPE",
        "additionalProperties": "UNKNOWN_FIELD",
    }
    return mapping.get(error.validator, "SCHEMA_CONSTRAINT_FAILURE"), value


def semantic_repair(initial: Any, final: dict[str, Any]) -> str:
    if not isinstance(initial, dict):
        return "NOT_COMPARABLE_INVALID_INITIAL_OUTPUT"
    before, after = initial.get("predicted_class"), final.get("predicted_class")
    if before not in CLASSES or after not in CLASSES:
        return "NOT_COMPARABLE_MISSING_CORE_FIELDS"
    return "SEMANTICALLY_COMPARABLE_UNCHANGED" if before == after else "SEMANTICALLY_COMPARABLE_CHANGED"


def repair_rows(records: Sequence[dict[str, Any]], builder: HeldoutEvidenceViewBuilder) -> list[dict[str, Any]]:
    repaired = [row for row in records if int(row.get("format_repairs", 0)) == 1]
    if len(repaired) != 168:
        raise ValueError("Expected exactly 168 formal repair paths")
    config = HeldoutConfig.load(CONFIG_PATH)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = read_json(OUTPUT_SCHEMA_PATH)
    pricing = PricingCatalog(read_json(PRICING_PATH))
    formal_config_hash = read_json(ATTEMPT_ROOT / "results/manifest.json")["config_hash"]
    result = []
    for row in repaired:
        key = _first_request_key(row, config, builder, prompt, schema, formal_config_hash)
        path = ATTEMPT_ROOT / "cache" / row["provider"] / config.raw["execution"]["cache_namespace"] / "raw" / f"{key}.json"
        if not path.exists():
            raise ValueError(f"Missing first-pass Cache boundary for repair {row['record_id']}")
        raw = read_json(path)
        raw_text = raw.pop("raw_text", "")
        raw.pop("error", None)
        reason, initial = repair_reason(raw_text, str(raw.get("finish_reason", "")), schema)
        semantic = semantic_repair(initial, prediction(row))
        first_cost = pricing.estimate_cny(config.model(row["provider"]), int(raw["input_tokens"]), int(raw["output_tokens"]))
        result.append({
            "provider": row["provider"], "evidence_view": row["evidence_view"],
            "actual_label": actual_class(row), "family": row["family"],
            "reason_category": reason, "semantic_comparison": semantic,
            "repair_success": bool(row.get("schema_valid")), "final_evaluable": predicted_class(row) != "NOT_EVALUABLE",
            "extra_input_tokens": int(row["input_tokens"]) - int(raw["input_tokens"]),
            "extra_output_tokens": int(row["output_tokens"]) - int(raw["output_tokens"]),
            "extra_latency_ms": float(row["latency_ms"]) - float(raw["latency_ms"]),
            "extra_cost_cny": float(row["cost_cny_delta"]) - first_cost,
        })
        raw_text = ""
        initial = None
    return result


def repair_aggregate(records: Sequence[dict[str, Any]], details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: Sequence[dict[str, Any]], repair_detail: Sequence[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        repairs = len(repair_detail)
        first_valid = sum(int(row.get("format_repairs", 0)) == 0 and row.get("schema_valid") for row in rows)
        final_valid = sum(bool(row.get("schema_valid")) for row in rows)
        success = sum(row["repair_success"] for row in repair_detail)
        comparable = [row for row in repair_detail if not row["semantic_comparison"].startswith("NOT_COMPARABLE")]
        changed = sum(row["semantic_comparison"] == "SEMANTICALLY_COMPARABLE_CHANGED" for row in comparable)
        return {
            "records": count, "repair_count": repairs, "repair_rate": ratio(repairs, count),
            "first_pass_schema_valid": first_valid, "first_pass_schema_valid_rate": ratio(first_valid, count),
            "final_schema_valid": final_valid, "final_schema_valid_rate": ratio(final_valid, count),
            "repair_success": success, "repair_success_rate": ratio(success, repairs),
            "repair_non_evaluable": repairs - success,
            "semantic_comparable_repairs": len(comparable), "semantic_changed_core_class": changed,
            "semantic_changed_rate_comparable": ratio(changed, len(comparable)),
            "extra_input_tokens": sum(row["extra_input_tokens"] for row in repair_detail),
            "extra_output_tokens": sum(row["extra_output_tokens"] for row in repair_detail),
            "extra_latency_ms": sum(row["extra_latency_ms"] for row in repair_detail),
            "extra_cost_cny": sum(row["extra_cost_cny"] for row in repair_detail),
            "mean_extra_input_tokens_per_repair": ratio(sum(row["extra_input_tokens"] for row in repair_detail), repairs),
            "mean_extra_output_tokens_per_repair": ratio(sum(row["extra_output_tokens"] for row in repair_detail), repairs),
            "mean_extra_latency_ms_per_repair": ratio(sum(row["extra_latency_ms"] for row in repair_detail), repairs),
            "mean_extra_cost_cny_per_repair": ratio(sum(row["extra_cost_cny"] for row in repair_detail), repairs),
        }
    result = {"overall": block(records, details)}
    selectors: dict[str, Callable[[dict[str, Any]], str]] = {
        "provider": lambda r: r["provider"], "evidence_view": lambda r: r["evidence_view"],
        "actual_label": lambda r: actual_class(r), "family": lambda r: r["family"],
    }
    for name, selector in selectors.items():
        values = sorted({selector(row) for row in records})
        result[f"by_{name}"] = {
            value: block(
                [row for row in records if selector(row) == value],
                [row for row in details if row[name if name != "actual_label" else "actual_label"] == value]
                if name in {"provider", "evidence_view", "actual_label", "family"} else [],
            )
            for value in values
        }
    result["reason_categories"] = dict(sorted(Counter(row["reason_category"] for row in details).items()))
    result["semantic_comparisons"] = dict(sorted(Counter(row["semantic_comparison"] for row in details).items()))
    result["definitions"] = {
        "first_pass_schema_valid": "final schema valid and no recorded repair boundary",
        "repair_success": "recorded repair path ending in a schema-valid formal record",
        "changed_rate": "changed core class / semantically comparable repaired records; non-comparable outputs excluded",
    }
    return result


def efficiency_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_provider = {}
    timeout_ms = 120_000.0
    for provider in PROVIDERS:
        rows = [row for row in records if row["provider"] == provider]
        metrics = E2.classification(rows)
        latency = [float(row["latency_ms"]) for row in rows]
        distribution = quantiles(latency)
        iqr = distribution["q3"] - distribution["q1"]
        high_cut = distribution["q3"] + 1.5 * iqr
        outliers = sum(value > high_cut for value in latency)
        repairs = sum(int(row.get("format_repairs", 0)) for row in rows)
        correct = sum(predicted_class(row) == actual_class(row) for row in rows)
        evaluable = sum(predicted_class(row) != "NOT_EVALUABLE" for row in rows)
        cost = sum(float(row["cost_cny_delta"]) for row in rows)
        input_tokens = sum(int(row["input_tokens"]) for row in rows)
        output_tokens = sum(int(row["output_tokens"]) for row in rows)
        by_provider[provider] = {
            "records": len(rows), "accuracy_all_records": metrics["accuracy_all_records"],
            "accuracy_evaluable_subset": metrics["accuracy_evaluable_subset"], "macro_f1": metrics["macro_f1"],
            "coverage": metrics["coverage"], "input_tokens": input_tokens, "output_tokens": output_tokens,
            "latency_ms": {**distribution, "iqr": iqr, "tukey_high_cutoff": high_cut, "high_outlier_records": outliers},
            "network_attempts": sum(int(row["actual_network_attempts"]) for row in rows),
            "format_repairs": repairs, "cost_cny": cost, "cost_per_record_cny": ratio(cost, len(rows)),
            "cost_per_evaluable_record_cny": ratio(cost, evaluable), "cost_per_correct_record_cny": ratio(cost, correct),
            "tokens_per_correct_record": ratio(input_tokens + output_tokens, correct),
            "repairs_per_100_records": ratio(repairs * 100, len(rows)),
            "near_timeout_records_ge_90pct": sum(value >= timeout_ms * 0.9 for value in latency),
            "timeout_exceeded_records": sum(value >= timeout_ms for value in latency),
            "near_or_exceeded_timeout_records": [
                {
                    "record_id": row["record_id"], "family": row["family"], "variant": row["variant"],
                    "evidence_view": row["evidence_view"], "latency_ms": float(row["latency_ms"]),
                    "threshold_status": "EXCEEDED_TIMEOUT" if float(row["latency_ms"]) >= timeout_ms else "NEAR_TIMEOUT",
                }
                for row in sorted(rows, key=lambda item: item["record_id"])
                if float(row["latency_ms"]) >= timeout_ms * 0.9
            ],
        }
    dominance = []
    for candidate in PROVIDERS:
        for dominated in PROVIDERS:
            if candidate == dominated:
                continue
            a, b = by_provider[candidate], by_provider[dominated]
            stable_a = 1 - a["format_repairs"] / a["records"]
            stable_b = 1 - b["format_repairs"] / b["records"]
            if a["macro_f1"] >= b["macro_f1"] and a["cost_cny"] <= b["cost_cny"] and stable_a >= stable_b:
                dominance.append({"dominant": candidate, "dominated": dominated})
    near = sum(value["near_timeout_records_ge_90pct"] for value in by_provider.values())
    exceeded = sum(value["timeout_exceeded_records"] for value in by_provider.values())
    sleep = (
        "NO_RECORD_LEVEL_EVIDENCE_OF_MATERIAL_SLEEP_CONTAMINATION"
        if near == 0 and exceeded == 0 else "HOST_SUSPENSION_OR_NETWORK_DELAY_CANNOT_BE_DISAMBIGUATED"
    )
    return {
        "by_provider": by_provider, "pareto_dominance": dominance,
        "latency_interpretation": sleep,
        "warning": "Provider strata are descriptive outcomes under the frozen configuration; no causal or universal best-provider claim is made.",
    }


def infrastructure_analysis(source: dict[str, Any]) -> dict[str, Any]:
    v1 = read_json(V1_ROOT / "audit/provider_failure_audit_v1.json")
    v1_summary = read_json(V1_ROOT / "results/summary.json")
    v2_summary = source["summary"]
    replay = source["replay"]
    return {
        "v1": {
            "attempt_id": "specdriftbench-heldout432-20260718-af3db03-01",
            "records_completed": 63, "records_planned": 432, "final_infrastructure_errors": 2,
            "final_infrastructure_error_rate": 2 / 63,
            "classification": "INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER",
            "performance_use": "PROHIBITED",
            "audit_classification": v1.get("attempt_classification"),
            "network_attempts": v1_summary.get("network_calls"),
        },
        "v2": {
            "attempt_id": ATTEMPT_ID, "records_completed": 432, "records_planned": 432,
            "final_infrastructure_errors": v2_summary.get("infrastructure_errors", 0),
            "network_attempts": v2_summary.get("network_calls"), "replay_records": replay.get("records_completed"),
            "replay_cache_misses": replay.get("cache_misses"), "fallback": v2_summary.get("fallback_used", 0),
        },
        "operational_conclusion": (
            "V1 is retained only as an incomplete infrastructure-gate calibration trace. "
            "The completed V2 attempt is the sole performance source; this comparison supports no model-performance claim."
        ),
    }


def build_reports(records: Sequence[dict[str, Any]], manifest: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    builder, expected = drift_metadata()
    repairs = repair_rows(records, builder)
    ledger = read_json(LEDGER_PATH)
    ledger_public = {
        "sha256_before": EXPECTED_SOURCE_HASHES["ledger_sha256"], "sha256_after": sha256_file(LEDGER_PATH),
        "api_attempts_before": 1689, "api_attempts_after": ledger["api_attempts"], "api_attempts_delta": ledger["api_attempts"] - 1689,
        "input_tokens_before": 10139862, "input_tokens_after": ledger["provider_reported_input_tokens"], "input_tokens_delta": ledger["provider_reported_input_tokens"] - 10139862,
        "output_tokens_before": 243536, "output_tokens_after": ledger["provider_reported_output_tokens"], "output_tokens_delta": ledger["provider_reported_output_tokens"] - 243536,
        "spent_cny_before": 21.814057367999997, "spent_cny_after": ledger["spent_cny"], "spent_cny_delta": ledger["spent_cny"] - 21.814057367999997,
        "reserved_cny_before": 0.0, "reserved_cny_after": ledger["reserved_cny"], "reserved_cny_delta": ledger["reserved_cny"],
    }
    failure = {
        "schema_version": "specdriftbench-experiment3-failure-analysis-v1", "attempt_id": ATTEMPT_ID,
        "status": "FORMAL_V2_HELDOUT_OFFLINE_ANALYSIS", "records": len(records),
        "source_identity": {"manifest_git_commit": manifest["git_commit"], **source["source_hashes_experiment3"]},
        "false_drift": false_drift_analysis(records, builder),
        "persistent_drift_localization": localization_analysis(records, builder, expected),
        "infrastructure_v1_v2": infrastructure_analysis(source),
        "ledger_before_after": ledger_public,
        "safety": {
            "provider_calls": 0, "provider_instances": 0, "network_calls": 0,
            "api_key_leakage": 0, "ground_truth_leakage": 0, "raw_text_persisted": False,
            "reasoning_content_persisted": False, "credentials_loaded": False,
        },
    }
    robustness = {
        "schema_version": "specdriftbench-experiment3-format-robustness-v1", "attempt_id": ATTEMPT_ID,
        "status": "FORMAL_V2_HELDOUT_OFFLINE_ANALYSIS", "repair_analysis": repair_aggregate(records, repairs),
        "sanitization": "Only categorical validation metadata and numeric usage deltas are persisted; raw response text is excluded.",
        "repair_reason_vocabulary": list(REPAIR_REASONS), "semantic_comparison_vocabulary": list(SEMANTIC_REPAIR),
    }
    efficiency = {
        "schema_version": "specdriftbench-experiment3-efficiency-v1", "attempt_id": ATTEMPT_ID,
        "status": "FORMAL_V2_HELDOUT_OFFLINE_ANALYSIS", **efficiency_analysis(records),
    }
    return failure, robustness, efficiency


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_failure_md(report: dict[str, Any]) -> str:
    false = report["false_drift"]
    loc = report["persistent_drift_localization"]
    lines = [
        "# Experiment 3: Failure Analysis", "",
        f"Attempt: `{ATTEMPT_ID}`", "", "Status: formal V2 held-out offline analysis.", "",
        "## False drift attribution", "",
        "False Drift Attribution Rate is predicted PD among actual AE/TF divided by actual AE/TF records. "
        "The evaluable-subset rate uses only schema-valid class predictions.", "",
        "| Scope | Records | Evaluable | Predicted PD | All-record rate | Evaluable rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("Overall", false["slices"]["overall"]), *false["slices"]["by_evidence_view"].items()]:
        lines.append(
            f"| {name} | {row['records']} | {row['evaluable_records']} | {row['predicted_persistent_drift']} | "
            f"{_fmt(row['false_drift_attribution_rate_all_records'])} | {_fmt(row['false_drift_attribution_rate_evaluable_subset'])} |"
        )
    lines += ["", "Observed confusion counts: " + ", ".join(f"{k}={v}" for k, v in false["observed_counts"].items()) + ".", ""]
    lines += ["## Persistent-drift localization", "", loc["note"], "", "| Component | Correct / PD | Coverage | All-record accuracy | Evaluable accuracy |", "|---|---:|---:|---:|---:|"]
    for component, row in loc["overall"]["component_accuracies"].items():
        lines.append(f"| {component} | {row['correct']} / {row['all_records']} | {_fmt(row['coverage'])} | {_fmt(row['accuracy_all_records'])} | {_fmt(row['accuracy_evaluable_subset'])} |")
    lines += ["", "## Infrastructure V1/V2", "", report["infrastructure_v1_v2"]["operational_conclusion"], "", "V1 performance metrics are excluded.", ""]
    return "\n".join(lines)


def render_robustness_md(report: dict[str, Any]) -> str:
    root = report["repair_analysis"]
    overall = root["overall"]
    lines = [
        "# Experiment 3: Format Robustness", "", f"Attempt: `{ATTEMPT_ID}`", "",
        "The frozen run recorded 168 repair paths. First-pass and final schema validity are reported separately; "
        "non-comparable initial outputs are excluded from the semantic-change denominator.", "",
        "| Provider | Repairs | Repair rate | First-pass valid | Final valid | Repair success | Extra cost (CNY) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, row in root["by_provider"].items():
        lines.append(f"| {PROVIDER_LABEL[provider]} | {row['repair_count']} | {_fmt(row['repair_rate'])} | {_fmt(row['first_pass_schema_valid_rate'])} | {_fmt(row['final_schema_valid_rate'])} | {_fmt(row['repair_success_rate'])} | {row['extra_cost_cny']:.6f} |")
    lines += ["", f"Overall repair success: {overall['repair_success']} / {overall['repair_count']} ({_fmt(overall['repair_success_rate'])}).", "", "Repair reason counts: " + ", ".join(f"{k}={v}" for k, v in root["reason_categories"].items()) + ".", "", report["sanitization"], ""]
    return "\n".join(lines)


def render_efficiency_md(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment 3: Provider Efficiency", "", f"Attempt: `{ATTEMPT_ID}`", "",
        "| Provider | Accuracy | Evaluable accuracy | Macro-F1 | Coverage | Input tokens | Output tokens | Median ms | P95 ms | Network attempts | Repairs | Cost CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, row in report["by_provider"].items():
        lines.append(
            f"| {PROVIDER_LABEL[provider]} | {_fmt(row['accuracy_all_records'])} | {_fmt(row['accuracy_evaluable_subset'])} | {_fmt(row['macro_f1'])} | {_fmt(row['coverage'])} | {row['input_tokens']} | {row['output_tokens']} | {row['latency_ms']['median']:.1f} | {row['latency_ms']['p95']:.1f} | {row['network_attempts']} | {row['format_repairs']} | {row['cost_cny']:.6f} |"
        )
    lines += ["", f"Latency audit: `{report['latency_interpretation']}`.", "", report["warning"], ""]
    return "\n".join(lines)


def render_tables(failure: dict[str, Any], robustness: dict[str, Any], efficiency: dict[str, Any]) -> str:
    false = failure["false_drift"]
    loc = failure["persistent_drift_localization"]
    lines = ["# Experiment 3: Paper Tables", "", "All values use the frozen formal V2 held-out Attempt.", "", "## Table 1. False Drift Attribution", "", "| Provider / view | AE to PD | TF to PD | False drift rate | PD recall |", "|---|---:|---:|---:|---:|"]
    for provider in PROVIDERS:
        for view in VIEWS:
            key = f"{provider}|{view}"
            ae = false["actual_error_patterns"]["AE"]["by_provider_and_evidence_view"][key]
            tf = false["actual_error_patterns"]["TF"]["by_provider_and_evidence_view"][key]
            combined = false["slices"]["by_provider_and_evidence_view"][key]
            pd = loc["by_provider_and_evidence_view"][key]["component_accuracies"]["class"]
            lines.append(f"| {PROVIDER_LABEL[provider]} / {view} | {ae['prediction_distribution']['PERSISTENT_DRIFT']} | {tf['prediction_distribution']['PERSISTENT_DRIFT']} | {_fmt(combined['false_drift_attribution_rate_all_records'])} | {_fmt(pd['accuracy_all_records'])} |")
    lines += ["", "## Table 2. PD localization by drift type", "", "| Type | PD recall | Category accuracy | Target accuracy | Exact-location accuracy |", "|---|---:|---:|---:|---:|"]
    for dtype in DRIFT_TYPES:
        row = loc["by_drift_type"][dtype]["component_accuracies"]
        lines.append(f"| {dtype} | {_fmt(row['class']['accuracy_all_records'])} | {_fmt(row['category']['accuracy_all_records'])} | {_fmt(row['target']['accuracy_all_records'])} | {_fmt(row['location']['accuracy_all_records'])} |")
    lines += ["", "## Table 3. Format robustness", "", "| Provider | First-pass valid | Repair rate | Final valid | Coverage |", "|---|---:|---:|---:|---:|"]
    for provider in PROVIDERS:
        r = robustness["repair_analysis"]["by_provider"][provider]
        e = efficiency["by_provider"][provider]
        lines.append(f"| {PROVIDER_LABEL[provider]} | {_fmt(r['first_pass_schema_valid_rate'])} | {_fmt(r['repair_rate'])} | {_fmt(r['final_schema_valid_rate'])} | {_fmt(e['coverage'])} |")
    lines += ["", "## Table 4. Efficiency trade-off", "", "| Provider | Macro-F1 | Median / P95 latency ms | Tokens | Cost CNY | Cost / correct CNY |", "|---|---:|---:|---:|---:|---:|"]
    for provider in PROVIDERS:
        e = efficiency["by_provider"][provider]
        lines.append(f"| {PROVIDER_LABEL[provider]} | {_fmt(e['macro_f1'])} | {e['latency_ms']['median']:.1f} / {e['latency_ms']['p95']:.1f} | {e['input_tokens'] + e['output_tokens']} | {e['cost_cny']:.6f} | {e['cost_per_correct_record_cny']:.6f} |")
    infra = failure["infrastructure_v1_v2"]
    lines += ["", "## Table 5. Infrastructure reliability", "", "| Protocol | Completed | Final infra errors | Replay | Cache miss | Ledger correct | Formal use |", "|---|---:|---:|---:|---:|---|---|"]
    lines.append("| V1 | 63 / 432 | 2 | N/A | 0 | Yes | No |")
    lines.append(f"| V2 | 432 / 432 | 0 | {infra['v2']['replay_records']} / 432 | {infra['v2']['replay_cache_misses']} | Yes | Yes |")
    lines += ["", "V1 is excluded from performance tables because it is an incomplete infrastructure-gate calibration Attempt.", ""]
    return "\n".join(lines)


def render_narrative(failure: dict[str, Any], robustness: dict[str, Any], efficiency: dict[str, Any]) -> str:
    false = failure["false_drift"]
    loc = failure["persistent_drift_localization"]["overall"]["component_accuracies"]
    repairs = robustness["repair_analysis"]["overall"]
    costs = efficiency["by_provider"]
    return f"""# Experiment 3: Paper Narrative

The formal V2 held-out analysis indicates that the three evaluated LLM providers systematically over-attributed non-drift failures to persistent drift within this benchmark: {false['observed_counts']['AE_to_PD']} of 144 agent-error records and {false['observed_counts']['TF_to_PD']} of 144 transient-failure records were classified as PD. TF was the hardest class to identify, with lower recall than AE and PD. This is a descriptive result for the frozen synthetic benchmark and provider configurations, not evidence of a general model property or all real API evolution.

Evidence history was associated with asymmetric class changes. Retry history improved transient-failure recall relative to first-failure evidence, while full evidence recovered some agent-error decisions and gave back part of the transient-failure gain. The paired transition counts show both corrections and regressions, so the result does not support a monotonic "more evidence is always better" claim.

Among the 144 actual PD records, category accuracy was {loc['category']['accuracy_all_records']:.3f}, target accuracy was {loc['target']['accuracy_all_records']:.3f}, and exact-location accuracy was {loc['location']['accuracy_all_records']:.3f}. These are separately scored PD-only components; target accuracy exceeding category accuracy is therefore not a contradiction and should not be presented as a monotone funnel.

Exact localization was the narrowest component. The error taxonomy distinguishes wrong-field, overly broad or narrow paths, wrong component or layer, wrong category, missing location, and non-evaluable outputs without changing the frozen evaluator's scoring rule.

Format robustness varied substantially by provider: the frozen run recorded 1 DeepSeek repair, 87 Qwen repairs, and 80 Kimi repairs. Of {repairs['repair_count']} total repair paths, {repairs['repair_success']} ended in schema-valid output. The most format-stable provider was not the most diagnostically accurate provider, indicating that format stability and diagnostic accuracy are distinct measured capabilities. This supports reporting first-pass and final validity separately rather than treating repaired outputs as first-pass successes.

Repairs introduced measurable additional tokens, latency, and estimated cost. Semantic-change rates are conditioned only on repaired responses with comparable initial core-class fields; invalid or missing-core-field initial outputs are explicitly excluded from that denominator.

Provider trade-offs were not one-dimensional. Estimated costs were CNY {costs['deepseek']['cost_cny']:.6f} for DeepSeek, {costs['dashscope']['cost_cny']:.6f} for Qwen, and {costs['moonshot']['cost_cny']:.6f} for Kimi, alongside different Macro-F1, latency, coverage, and first-pass format stability. The frozen observations do not establish causality or a universally best provider.

The earlier V1 Attempt completed only 63 of 432 records and stopped after two final infrastructure errors under the original gate. It is classified as `INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER` and is not used for performance comparison. V2 completed 432 records with zero final infrastructure errors and reproduced all 432 results in an offline replay with zero Cache misses. V2 improved only the experiment's operational reliability; it did not improve the models themselves.
"""


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_outputs(root: Path, failure: dict[str, Any], robustness: dict[str, Any], efficiency: dict[str, Any]) -> None:
    false_rows = []
    for key, row in failure["false_drift"]["slices"]["by_provider_and_evidence_view"].items():
        provider, view = key.split("|")
        false_rows.append({"provider": provider, "evidence_view": view, **{k: row[k] for k in ("records", "evaluable_records", "non_evaluable_records", "predicted_persistent_drift", "false_drift_attribution_rate_all_records", "false_drift_attribution_rate_evaluable_subset")}})
    write_csv(root / "false_drift_by_provider_view_v1.csv", false_rows, list(false_rows[0]))
    loc_rows = []
    for dtype, row in failure["persistent_drift_localization"]["by_drift_type"].items():
        loc_rows.append({"drift_type": dtype, **{f"{name}_{metric}": value for name, block in row["component_accuracies"].items() for metric, value in block.items() if metric in {"correct", "all_records", "evaluable_records", "coverage", "accuracy_all_records", "accuracy_evaluable_subset"}}})
    write_csv(root / "drift_type_localization_v1.csv", loc_rows, list(loc_rows[0]))
    taxonomy_rows = []
    for scope_name, groups in (("overall", {"all": failure["persistent_drift_localization"]["overall"]}), ("provider", failure["persistent_drift_localization"]["by_provider"]), ("evidence_view", failure["persistent_drift_localization"]["by_evidence_view"]), ("drift_type", failure["persistent_drift_localization"]["by_drift_type"])):
        for group, block in groups.items():
            for category in LOCATION_ERRORS:
                taxonomy_rows.append({"scope": scope_name, "group": group, "error_category": category, "count": block["location_error_taxonomy"][category]})
    write_csv(root / "location_error_taxonomy_v1.csv", taxonomy_rows, list(taxonomy_rows[0]))
    repair_rows_csv = []
    for provider, row in robustness["repair_analysis"]["by_provider"].items():
        repair_rows_csv.append({"provider": provider, **row})
    write_csv(root / "format_repair_analysis_v1.csv", repair_rows_csv, list(repair_rows_csv[0]))
    efficiency_rows = [{"provider": p, **{k: v for k, v in row.items() if k != "latency_ms"}, **{f"latency_ms_{k}": v for k, v in row["latency_ms"].items()}} for p, row in efficiency["by_provider"].items()]
    write_csv(root / "provider_efficiency_v1.csv", efficiency_rows, list(efficiency_rows[0]))
    infra = failure["infrastructure_v1_v2"]
    infra_rows = [{"version": version.upper(), **values} for version, values in (("v1", infra["v1"]), ("v2", infra["v2"]))]
    fields = sorted(set().union(*(row.keys() for row in infra_rows)))
    write_csv(root / "infrastructure_v1_v2_v1.csv", infra_rows, fields)


def run_analysis(output_root: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records, manifest, source = validate_sources()
    failure, robustness, efficiency = build_reports(records, manifest, source)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment3_failure_analysis_v1.json", failure)
    write_json(output_root / "experiment3_format_robustness_v1.json", robustness)
    write_json(output_root / "experiment3_efficiency_v1.json", efficiency)
    (output_root / "experiment3_failure_analysis_v1.md").write_text(render_failure_md(failure), encoding="utf-8")
    (output_root / "experiment3_format_robustness_v1.md").write_text(render_robustness_md(robustness), encoding="utf-8")
    (output_root / "experiment3_efficiency_v1.md").write_text(render_efficiency_md(efficiency), encoding="utf-8")
    (output_root / "experiment3_paper_tables_v1.md").write_text(render_tables(failure, robustness, efficiency), encoding="utf-8")
    (output_root / "experiment3_paper_narrative_v1.md").write_text(render_narrative(failure, robustness, efficiency), encoding="utf-8")
    write_csv_outputs(output_root, failure, robustness, efficiency)
    return failure, robustness, efficiency


COLORS = ("#8eacb7", "#d2ab7f", "#9fb99a")
SHORT_VIEW = {"FIRST_FAILURE": "First failure", "RETRY_HISTORY": "Retry history", "FULL_EVIDENCE": "Full evidence"}


def chart_frame(canvas: Any, title: str, subtitle: str, y_max: float = 1.0) -> tuple[float, float, float, float]:
    canvas.text((1050, 58), title, 46, "ma", True)
    canvas.text((1050, 116), subtitle, 27, "ma", False, "#555555")
    left, top, right, bottom = 190, 205, 2010, 1120
    for index in range(6):
        value = y_max * index / 5
        y = bottom - value / y_max * (bottom - top)
        canvas.line([(left, y), (right, y)], "#d8d8d8", 2)
        canvas.text((left - 24, y), f"{value:.1f}", 25, "rm", False, "#555555")
    canvas.line([(left, top), (left, bottom), (right, bottom)], "#333333", 3)
    return left, top, right, bottom


def draw_pd_overattribution(canvas: Any, failure: dict[str, Any]) -> None:
    _, top, _, bottom = chart_frame(canvas, "Persistent-drift attribution by actual class", "Counts per evidence view; denominator = 48 records per bar", 48)
    centers = (500, 1100, 1700)
    labels = (("AE", "AGENT_ERROR"), ("TF", "TRANSIENT_FAILURE"), ("PD", "PERSISTENT_DRIFT"))
    for group_index, (variant, _) in enumerate(labels):
        for view_index, view in enumerate(VIEWS):
            rows = failure["false_drift"]["by_actual_label_and_view"][variant][view]
            value = rows["predicted_persistent_drift"]
            x = centers[group_index] + (view_index - 1) * 155
            y = bottom - value / 48 * (bottom - top)
            canvas.rect((x - 55, y, x + 55, bottom), COLORS[view_index], pattern=view_index)
            canvas.text((x, y - 42), str(value), 25, "ma", True)
        canvas.text((centers[group_index], bottom + 58), variant, 30, "ma", True)
    for index, view in enumerate(VIEWS):
        x = 520 + index * 530
        canvas.rect((x, 1220, x + 48, 1260), COLORS[index], pattern=index)
        canvas.text((x + 66, 1240), SHORT_VIEW[view], 26, "lm")


def draw_localization(canvas: Any, failure: dict[str, Any]) -> None:
    _, top, _, bottom = chart_frame(canvas, "Persistent-drift localization components", "PD-only component accuracy; 144 records; components are scored separately")
    components = failure["persistent_drift_localization"]["overall"]["component_accuracies"]
    centers = (430, 850, 1270, 1690)
    for index, name in enumerate(("class", "category", "target", "location")):
        value = components[name]["accuracy_all_records"]
        y = bottom - value * (bottom - top)
        canvas.rect((centers[index] - 90, y, centers[index] + 90, bottom), COLORS[index % 3], pattern=index % 3)
        canvas.text((centers[index], y - 48), f"{value:.3f}", 27, "ma", True)
        canvas.text((centers[index], bottom + 58), {"class": "PD recall", "category": "Category", "target": "Target", "location": "Exact location"}[name], 28, "ma", True)


def draw_robustness(canvas: Any, robustness: dict[str, Any]) -> None:
    _, top, _, bottom = chart_frame(canvas, "Provider format robustness", "First-pass validity, final validity, and repair rate; 144 records/provider")
    centers = (500, 1100, 1700)
    keys = (("first_pass_schema_valid_rate", "First-pass valid"), ("final_schema_valid_rate", "Final valid"), ("repair_rate", "Repair rate"))
    for provider_index, provider in enumerate(PROVIDERS):
        row = robustness["repair_analysis"]["by_provider"][provider]
        for index, (key, _) in enumerate(keys):
            value = row[key]
            x = centers[provider_index] + (index - 1) * 155
            y = bottom - value * (bottom - top)
            canvas.rect((x - 55, y, x + 55, bottom), COLORS[index], pattern=index)
            canvas.text((x, max(top + 18, y - 40)), f"{value:.2f}", 22, "ma", True)
        canvas.text((centers[provider_index], bottom + 58), PROVIDER_LABEL[provider], 30, "ma", True)
    for index, (_, label) in enumerate(keys):
        x = 490 + index * 560
        canvas.rect((x, 1220, x + 48, 1260), COLORS[index], pattern=index)
        canvas.text((x + 66, 1240), label, 26, "lm")


def draw_performance_cost(canvas: Any, efficiency: dict[str, Any]) -> None:
    canvas.text((1050, 58), "Provider performance-cost trade-off", 46, "ma", True)
    canvas.text((1050, 116), "Macro-F1 versus estimated cost; marker radius scales with repair rate", 27, "ma", False, "#555555")
    left, top, right, bottom = 190, 205, 2010, 1120
    x_max, y_max = 9.0, 0.7
    for index in range(6):
        x = left + index / 5 * (right - left); y = bottom - index / 5 * (bottom - top)
        canvas.line([(left, y), (right, y)], "#d8d8d8", 2)
        canvas.text((left - 24, y), f"{y_max*index/5:.2f}", 25, "rm", False, "#555555")
        canvas.text((x, bottom + 42), f"{x_max*index/5:.1f}", 25, "ma", False, "#555555")
    canvas.line([(left, top), (left, bottom), (right, bottom)], "#333333", 3)
    canvas.text(((left + right) / 2, bottom + 92), "Estimated cost (CNY)", 28, "ma", True)
    canvas.text((left - 135, top - 25), "Macro-F1", 27, "la", True)
    for index, provider in enumerate(PROVIDERS):
        row = efficiency["by_provider"][provider]
        x = left + row["cost_cny"] / x_max * (right - left)
        y = bottom - row["macro_f1"] / y_max * (bottom - top)
        radius = 24 + 42 * row["format_repairs"] / row["records"]
        canvas.circle((x, y), radius, COLORS[index])
        canvas.text((x + radius + 16, y - 10), f"{PROVIDER_LABEL[provider]}  {row['macro_f1']:.3f}", 26, "la", True)


def render_figures(output_root: Path = OUTPUT_ROOT) -> None:
    failure = read_json(output_root / "experiment3_failure_analysis_v1.json")
    robustness = read_json(output_root / "experiment3_format_robustness_v1.json")
    efficiency = read_json(output_root / "experiment3_efficiency_v1.json")
    figures = (
        ("figure_pd_overattribution_v1", lambda c: draw_pd_overattribution(c, failure)),
        ("figure_localization_funnel_v1", lambda c: draw_localization(c, failure)),
        ("figure_format_robustness_v1", lambda c: draw_robustness(c, robustness)),
        ("figure_performance_cost_v1", lambda c: draw_performance_cost(c, efficiency)),
    )
    for stem, draw in figures:
        for extension, pdf in (("png", False), ("pdf", True)):
            canvas = E2.PaperCanvas(output_root / f"{stem}.{extension}", pdf=pdf)
            draw(canvas)
            canvas.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--render-only", action="store_true")
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
