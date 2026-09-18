from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/analyze_specdriftbench_experiment3.py"
spec = importlib.util.spec_from_file_location("experiment3_analysis", SCRIPT)
assert spec is not None and spec.loader is not None
E3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E3)


def load_outputs():
    root = E3.OUTPUT_ROOT
    return (
        json.loads((root / "experiment3_failure_analysis_v1.json").read_text()),
        json.loads((root / "experiment3_format_robustness_v1.json").read_text()),
        json.loads((root / "experiment3_efficiency_v1.json").read_text()),
    )


@pytest.fixture(scope="module")
def records():
    return E3.E2.load_and_validate(E3.ATTEMPT_ROOT)[0]


@pytest.fixture(scope="module")
def outputs():
    return load_outputs()


def test_record_count(records):
    assert len(records) == 432


def test_evaluable_count(records):
    assert sum(E3.predicted_class(row) != "NOT_EVALUABLE" for row in records) == 427


def test_non_evaluable_count(records):
    assert sum(E3.predicted_class(row) == "NOT_EVALUABLE" for row in records) == 5


@pytest.mark.parametrize("variant", ["AE", "TF", "PD"])
def test_variant_balance(records, variant):
    assert sum(row["variant"] == variant for row in records) == 144


def test_family_balance(records):
    assert len({row["family"] for row in records}) == 16


@pytest.mark.parametrize("provider", E3.PROVIDERS)
def test_provider_balance(records, provider):
    assert sum(row["provider"] == provider for row in records) == 144


@pytest.mark.parametrize("view", E3.VIEWS)
def test_view_balance(records, view):
    assert sum(row["evidence_view"] == view for row in records) == 144


def test_no_development_overlap(records):
    assert not ({row["family"] for row in records} & {"M01", "M06", "M11", "M16"})


def test_response_availability(records):
    assert sum(row.get("content_length") is not None for row in records) == 432


@pytest.mark.parametrize("drift_type", E3.DRIFT_TYPES)
def test_four_families_per_drift_type(records, drift_type):
    builder = E3.HeldoutEvidenceViewBuilder()
    families = {row["family"] for row in records if E3.drift_type(row["family"], builder) == drift_type}
    assert len(families) == 4


@pytest.mark.parametrize("drift_type", E3.DRIFT_TYPES)
def test_36_pd_records_per_drift_type(outputs, drift_type):
    failure, _, _ = outputs
    assert failure["persistent_drift_localization"]["by_drift_type"][drift_type]["records"] == 36


@pytest.mark.parametrize("key,expected", [("AE_to_PD", 80), ("TF_to_PD", 114), ("PD_to_PD", 135)])
def test_pd_confusion_counts(outputs, key, expected):
    failure, _, _ = outputs
    assert failure["false_drift"]["observed_counts"][key] == expected


def test_false_drift_denominator(outputs):
    failure, _, _ = outputs
    metric = failure["false_drift"]["slices"]["overall"]
    assert metric["records"] == 288
    assert metric["predicted_persistent_drift"] == 194
    assert metric["false_drift_attribution_rate_all_records"] == pytest.approx(194 / 288)


def test_false_drift_evaluable_denominator(outputs):
    failure, _, _ = outputs
    metric = failure["false_drift"]["slices"]["overall"]
    assert metric["evaluable_records"] == 283
    assert metric["false_drift_attribution_rate_evaluable_subset"] == pytest.approx(194 / 283)


def test_non_evaluable_never_correct(records):
    assert all(E3.predicted_class(row) != E3.actual_class(row) for row in records if E3.predicted_class(row) == "NOT_EVALUABLE")


def test_pd_prediction_metrics(outputs):
    failure, _, _ = outputs
    metric = failure["false_drift"]["pd_prediction_metrics"]
    assert metric["precision"] == pytest.approx(135 / 329)
    assert metric["recall"] == pytest.approx(135 / 144)
    assert metric["frequency"] == pytest.approx(329 / 432)


@pytest.mark.parametrize("component,correct", [("category", 135), ("target", 144), ("location", 79)])
def test_pd_localization_denominator_and_correct(outputs, component, correct):
    failure, _, _ = outputs
    metric = failure["persistent_drift_localization"]["overall"]["component_accuracies"][component]
    assert metric["all_records"] == metric["evaluable_records"] == 144
    assert metric["correct"] == correct
    assert metric["coverage"] == 1.0


def test_location_taxonomy_partition(outputs):
    failure, _, _ = outputs
    taxonomy = failure["persistent_drift_localization"]["overall"]["location_error_taxonomy"]
    assert set(taxonomy) == set(E3.LOCATION_ERRORS)
    assert sum(taxonomy.values()) == 144
    assert taxonomy["EXACT_MATCH"] == 79


def test_format_repair_total(records, outputs):
    _, robustness, _ = outputs
    assert sum(row["format_repairs"] for row in records) == 168
    assert robustness["repair_analysis"]["overall"]["repair_count"] == 168


@pytest.mark.parametrize("provider,expected", [("deepseek", 1), ("dashscope", 87), ("moonshot", 80)])
def test_provider_repair_count(outputs, provider, expected):
    _, robustness, _ = outputs
    assert robustness["repair_analysis"]["by_provider"][provider]["repair_count"] == expected


def test_first_pass_and_final_validity_separate(outputs):
    _, robustness, _ = outputs
    metric = robustness["repair_analysis"]["overall"]
    assert metric["first_pass_schema_valid"] == 263
    assert metric["final_schema_valid"] == 427
    assert metric["first_pass_schema_valid_rate"] != metric["final_schema_valid_rate"]


def test_repair_success_and_non_evaluable(outputs):
    _, robustness, _ = outputs
    metric = robustness["repair_analysis"]["overall"]
    assert metric["repair_success"] == 164
    assert metric["repair_non_evaluable"] == 4


def test_semantic_comparison_denominator(outputs):
    _, robustness, _ = outputs
    metric = robustness["repair_analysis"]["overall"]
    assert metric["semantic_comparable_repairs"] == 164
    assert metric["semantic_changed_core_class"] == 8
    assert metric["semantic_changed_rate_comparable"] == pytest.approx(8 / 164)


def test_not_comparable_excluded(outputs):
    _, robustness, _ = outputs
    counts = robustness["repair_analysis"]["semantic_comparisons"]
    assert counts["NOT_COMPARABLE_MISSING_CORE_FIELDS"] == 4
    assert sum(counts.values()) == 168
    assert sum(v for k, v in counts.items() if k.startswith("SEMANTICALLY_COMPARABLE")) == 164


def test_repair_reason_vocabulary(outputs):
    _, robustness, _ = outputs
    assert set(robustness["repair_reason_vocabulary"]) == set(E3.REPAIR_REASONS)
    assert robustness["repair_analysis"]["reason_categories"] == {"SCHEMA_CONSTRAINT_FAILURE": 168}


@pytest.mark.parametrize(
    "provider,cost", [("deepseek", 1.248579444), ("dashscope", 4.23168), ("moonshot", 8.29835165)]
)
def test_provider_cost(outputs, provider, cost):
    _, _, efficiency = outputs
    assert efficiency["by_provider"][provider]["cost_cny"] == pytest.approx(cost)


def test_cost_per_correct(outputs):
    _, _, efficiency = outputs
    for row in efficiency["by_provider"].values():
        correct = row["accuracy_all_records"] * row["records"]
        assert row["cost_per_correct_record_cny"] == pytest.approx(row["cost_cny"] / correct)


def test_latency_records_not_deleted(records, outputs):
    _, _, efficiency = outputs
    assert sum(row["records"] for row in efficiency["by_provider"].values()) == len(records)
    assert efficiency["by_provider"]["deepseek"]["latency_ms"]["max"] == max(
        row["latency_ms"] for row in records if row["provider"] == "deepseek"
    )


def test_latency_ambiguity_marker(outputs):
    _, _, efficiency = outputs
    assert efficiency["latency_interpretation"] == "HOST_SUSPENSION_OR_NETWORK_DELAY_CANNOT_BE_DISAMBIGUATED"
    assert efficiency["by_provider"]["deepseek"]["timeout_exceeded_records"] == 2


def test_v1_excluded_from_performance(outputs):
    failure, _, _ = outputs
    v1 = failure["infrastructure_v1_v2"]["v1"]
    assert v1["records_completed"] == 63
    assert v1["performance_use"] == "PROHIBITED"
    assert v1["classification"] == "INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER"


def test_v2_complete_and_replayed(outputs):
    failure, _, _ = outputs
    v2 = failure["infrastructure_v1_v2"]["v2"]
    assert v2["records_completed"] == v2["replay_records"] == 432
    assert v2["replay_cache_misses"] == v2["final_infrastructure_errors"] == 0


@pytest.mark.parametrize("key", E3.EXPECTED_SOURCE_HASHES)
def test_source_hashes_unchanged(key):
    observed = {
        "records_tree_sha256": E3.tree_hash(E3.ATTEMPT_ROOT / "results/records"),
        "cache_tree_sha256": E3.tree_hash(E3.ATTEMPT_ROOT / "cache"),
        "experiment2_tree_sha256": E3.tree_hash(E3.EXPERIMENT2_ROOT),
        "manifest_sha256": E3.sha256_file(E3.ATTEMPT_ROOT / "results/manifest.json"),
        "summary_sha256": E3.sha256_file(E3.ATTEMPT_ROOT / "results/summary.json"),
        "ledger_sha256": E3.sha256_file(E3.LEDGER_PATH),
    }
    assert observed[key] == E3.EXPECTED_SOURCE_HASHES[key]


def test_ledger_unchanged(outputs):
    failure, _, _ = outputs
    ledger = failure["ledger_before_after"]
    assert ledger["sha256_before"] == ledger["sha256_after"]
    assert ledger["api_attempts_delta"] == ledger["input_tokens_delta"] == ledger["output_tokens_delta"] == 0
    assert ledger["spent_cny_delta"] == ledger["reserved_cny_delta"] == 0


def test_offline_safety_counters(outputs):
    failure, _, _ = outputs
    safety = failure["safety"]
    assert safety["network_calls"] == safety["provider_calls"] == safety["provider_instances"] == 0
    assert safety["api_key_leakage"] == safety["ground_truth_leakage"] == 0


def test_no_sensitive_text_in_outputs():
    forbidden = ("Authorization:", "Bearer ", "sk-", "chain-of-thought")
    for path in E3.OUTPUT_ROOT.iterdir():
        if path.suffix.lower() in {".json", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_analysis_output_determinism_and_no_network(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is prohibited in Experiment 3")

    monkeypatch.setattr(socket, "create_connection", blocked)
    first, second = tmp_path / "first", tmp_path / "second"
    E3.run_analysis(first)
    E3.run_analysis(second)
    assert directory_hash(first) == directory_hash(second)


@pytest.mark.parametrize(
    "name",
    [
        "figure_pd_overattribution_v1", "figure_localization_funnel_v1",
        "figure_format_robustness_v1", "figure_performance_cost_v1",
    ],
)
def test_figure_files_exist_and_nonempty(name):
    png, pdf = E3.OUTPUT_ROOT / f"{name}.png", E3.OUTPUT_ROOT / f"{name}.pdf"
    assert png.stat().st_size > 20_000 and png.read_bytes().startswith(b"\x89PNG")
    assert pdf.stat().st_size > 1_000 and pdf.read_bytes().startswith(b"%PDF")


def test_exact_output_inventory():
    required = {
        "experiment3_failure_analysis_v1.json", "experiment3_failure_analysis_v1.md",
        "experiment3_format_robustness_v1.json", "experiment3_format_robustness_v1.md",
        "experiment3_efficiency_v1.json", "experiment3_efficiency_v1.md",
        "experiment3_paper_tables_v1.md", "experiment3_paper_narrative_v1.md",
        "false_drift_by_provider_view_v1.csv", "drift_type_localization_v1.csv",
        "location_error_taxonomy_v1.csv", "format_repair_analysis_v1.csv",
        "provider_efficiency_v1.csv", "infrastructure_v1_v2_v1.csv",
    }
    assert required <= {path.name for path in E3.OUTPUT_ROOT.iterdir()}


def test_paper_narrative_guardrails():
    text = (E3.OUTPUT_ROOT / "experiment3_paper_narrative_v1.md").read_text().lower()
    assert " prove" not in text
    assert "not evidence of a general model property" in text
    assert "not establish causality" in text
    assert "v1" in text and "v2" in text
