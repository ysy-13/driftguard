from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_specdriftbench_experiment2.py"
SPEC = importlib.util.spec_from_file_location("specdriftbench_experiment2", SCRIPT)
assert SPEC and SPEC.loader
E2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E2)


@pytest.fixture(scope="module")
def source():
    return E2.load_and_validate()


@pytest.fixture(scope="module")
def report(source):
    records, manifest, metadata = source
    return E2.build_report(records, manifest, metadata)[0]


def test_only_formal_v2_attempt_is_admissible(source):
    _, manifest, metadata = source
    wrong = copy.deepcopy(manifest)
    wrong["attempt_id"] = "specdriftbench-heldout432-20260718-af3db03-01"
    with pytest.raises(ValueError, match="formal V2"):
        E2.validate_source_identity(wrong, metadata["frozen"])


def test_canary_attempt_is_rejected(source):
    _, manifest, metadata = source
    wrong = copy.deepcopy(manifest)
    wrong["attempt_id"] = "specdriftbench-py312fixed-20260718-canary36-1"
    with pytest.raises(ValueError, match="formal V2"):
        E2.validate_source_identity(wrong, metadata["frozen"])


def test_exactly_432_records(source):
    records, _, _ = source
    assert len(records) == 432
    assert len({row["record_id"] for row in records}) == 432


def test_balanced_provider_view_and_variant_counts(source):
    records, _, _ = source
    assert Counter(row["provider"] for row in records) == Counter({p: 144 for p in E2.PROVIDERS})
    assert Counter(row["evidence_view"] for row in records) == Counter({v: 144 for v in E2.VIEWS})
    assert Counter(row["variant"] for row in records) == Counter({"AE": 144, "TF": 144, "PD": 144})


def test_no_development_family_enters_formal_source(source):
    records, manifest, _ = source
    assert set(row["family"] for row in records).isdisjoint(manifest["development_families"])
    assert len(set(row["family"] for row in records)) == 16


def test_144_complete_paired_groups(source):
    groups = E2.paired_rows(source[0])
    assert len(groups) == 144
    assert all(set(group) == set(E2.VIEWS) for group in groups.values())


def test_source_bytes_match_preflight_hashes(source):
    assert source[2]["source_hashes"] == E2.EXPECTED_SOURCE_HASHES


def test_frozen_fingerprints_and_analysis_plan(source):
    manifest = source[1]
    for key, expected in E2.EXPECTED_FINGERPRINTS.items():
        assert manifest["frozen_fingerprints"][key] == expected
    assert manifest["analysis_plan_sha256"] == E2.EXPECTED_ANALYSIS_PLAN


@pytest.mark.parametrize("view", E2.VIEWS)
def test_evidence_view_macro_f1_reproduces_frozen_analysis(report, view):
    assert report["by_evidence_view"][view]["macro_f1"] == pytest.approx(E2.EXPECTED_VIEW_MACRO_F1[view], abs=1e-15)


@pytest.mark.parametrize("view", E2.VIEWS)
def test_confusion_matrix_and_denominators_are_complete(report, view):
    metric = report["by_evidence_view"][view]
    assert sum(sum(row.values()) for row in metric["confusion_matrix"].values()) == 144
    assert metric["coverage"] == metric["evaluable_records"] / metric["records"]
    correct = sum(metric["confusion_matrix"][label][label] for label in E2.CLASSES)
    assert metric["accuracy_all_records"] == correct / 144
    assert metric["accuracy_evaluable_subset"] == correct / metric["evaluable_records"]


def test_missing_predictions_are_never_counted_correct(report):
    for metric in report["by_evidence_view"].values():
        non_evaluable = sum(metric["confusion_matrix"][label]["NOT_EVALUABLE"] for label in E2.CLASSES)
        assert metric["records"] - metric["evaluable_records"] == non_evaluable


@pytest.mark.parametrize("name", E2.EXPECTED_PAIRED_CI)
def test_frozen_family_clustered_paired_intervals(report, name):
    assert report["paired_comparisons"][name]["ci95_family_clustered_bootstrap"] == E2.EXPECTED_PAIRED_CI[name]


def test_paired_transition_counts_partition_all_groups(report):
    for comparison in report["paired_comparisons"].values():
        assert sum(comparison["transitions"].values()) == 144


def test_paired_delta_equals_net_transition_rate(report):
    for comparison in report["paired_comparisons"].values():
        transitions = comparison["transitions"]
        expected = (transitions["a_correct_b_incorrect"] - transitions["a_incorrect_b_correct"]) / 144
        assert comparison["accuracy_delta_all_records"] == expected


def test_joint_evaluable_sensitivity_has_explicit_denominator(report):
    for comparison in report["paired_comparisons"].values():
        assert 0 < comparison["jointly_evaluable"] <= 144
        assert comparison["jointly_evaluable_rate"] == comparison["jointly_evaluable"] / 144


def test_bootstrap_keeps_family_clusters_intact_and_uses_frozen_settings(report):
    settings = report["bootstrap"]["settings"]
    assert settings == {
        "unit": "family",
        "families_per_replicate": 16,
        "repetitions": 10_000,
        "seed": 20_260_718,
        "confidence_level": 0.95,
        "interval": "percentile",
    }


def test_bootstrap_is_deterministic(source):
    first = E2.bootstrap_statistics(source[0], repetitions=200, seed=E2.BOOTSTRAP_SEED)
    second = E2.bootstrap_statistics(source[0], repetitions=200, seed=E2.BOOTSTRAP_SEED)
    assert first == second


def test_provider_by_view_slices_are_48_records(report):
    cells = report["by_provider_and_evidence_view"]
    assert len(cells) == 3
    assert all(cells[provider][view]["records"] == 48 for provider in E2.PROVIDERS for view in E2.VIEWS)


def test_label_by_view_slices_partition_outcomes(report):
    for label in E2.CLASSES:
        for view in E2.VIEWS:
            metric = report["by_label_and_evidence_view"][label][view]
            assert metric["records"] == 48
            assert metric["correct"] + metric["incorrect"] + metric["non_evaluable"] == 48


def test_uniform_random_is_analytical_not_sampled(report):
    baseline = report["baselines"]["uniform_random"]
    assert baseline["label"] == "ANALYTICAL_EXPECTED_BASELINE"
    assert baseline["samples_drawn"] == 0
    assert all(baseline[name] == 1 / 3 for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"))


def test_always_pd_baseline_is_computed_with_expected_balanced_metrics(report):
    baseline = report["baselines"]["always_persistent_drift"]
    assert baseline["accuracy_all_records"] == 1 / 3
    assert baseline["macro_precision"] == 1 / 9
    assert baseline["macro_recall"] == 1 / 3
    assert baseline["macro_f1"] == 1 / 6
    assert baseline["per_class"]["PERSISTENT_DRIFT"]["f1"] == 0.5


def test_symbolic_oracle_is_fail_closed_as_non_comparable(report):
    oracle = report["baselines"]["symbolic_oracle"]
    assert oracle["value"] is None
    assert oracle["display"] == "N/A"
    assert oracle["reason"] == "protocol alignment not proven"
    assert oracle["label"] == "NON_COMPARABLE_SYMBOLIC_ORACLE_UPPER_BOUND"
    assert oracle["audit"]["phase7_result_count"] == 60


def test_no_api_provider_network_cache_or_credentials_are_used(report):
    assert report["safety"] == {
        "api_calls": 0,
        "provider_instances": 0,
        "network_calls": 0,
        "cache_reads": 0,
        "credentials_read": 0,
        "ground_truth_exposed_to_provider": False,
    }


def test_quality_separates_structured_output_failures_from_infrastructure(report):
    quality = report["quality"]
    assert quality["schema_valid"] == 427
    assert quality["invalid_structured_output"] == 4
    assert quality["output_truncated"] == 1
    assert quality["infrastructure_errors"] == 0
    assert quality["fallbacks"] == quality["leakage_failures"] == 0


def test_script_has_no_provider_or_environment_access_imports():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("import requests", "import httpx", "import socket", "load_dotenv", "os.environ", "provider_client")
    assert not any(token in source for token in forbidden)


def test_analysis_rerun_is_byte_identical(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    E2.run_analysis(first)
    E2.run_analysis(second)
    names = {
        "experiment2_evidence_ablation_v1.json",
        "experiment2_evidence_ablation_v1.md",
        "experiment2_baselines_v1.json",
        "experiment2_baselines_v1.md",
        "experiment2_paper_tables_v1.md",
        "experiment2_provider_view_metrics_v1.csv",
        "experiment2_paired_transitions_v1.csv",
        "experiment2_label_view_metrics_v1.csv",
    }
    for name in names:
        assert hashlib.sha256((first / name).read_bytes()).digest() == hashlib.sha256((second / name).read_bytes()).digest()


def test_transition_csv_contains_exactly_three_times_144_data_rows(tmp_path):
    E2.run_analysis(tmp_path)
    with (tmp_path / "experiment2_paired_transitions_v1.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 144


def test_formal_records_and_ledger_remain_byte_identical(source):
    assert E2.tree_hash(E2.ATTEMPT_ROOT / "results/records") == E2.EXPECTED_SOURCE_HASHES["records_tree_sha256"]
    report = json.loads((E2.OUTPUT_ROOT / "experiment2_evidence_ablation_v1.json").read_text(encoding="utf-8"))
    assert E2.sha256_file(E2.LEDGER_PATH) == report["source_integrity"]["ledger_sha256_at_analysis_start"]


def test_public_figures_exist_without_record_identifiers(source):
    record_ids = {row["record_id"] for row in source[0]}
    for stem in ("figure_evidence_performance_v1", "figure_paired_evidence_gain_v1", "figure_provider_evidence_v1"):
        for extension in ("png", "pdf"):
            path = E2.OUTPUT_ROOT / f"{stem}.{extension}"
            assert path.is_file() and path.stat().st_size > 1_000
            payload = path.read_bytes()
            assert not any(record_id.encode() in payload for record_id in record_ids)


def test_accuracy_and_macro_f1_divergence_is_evidenced(report):
    retry = report["by_evidence_view"]["RETRY_HISTORY"]
    full = report["by_evidence_view"]["FULL_EVIDENCE"]
    assert full["accuracy_all_records"] > retry["accuracy_all_records"]
    assert retry["macro_f1"] > full["macro_f1"]
    assert retry["per_class"]["TRANSIENT_FAILURE"]["recall"] > full["per_class"]["TRANSIENT_FAILURE"]["recall"]


def test_completion_status_is_experiment2_complete(report):
    assert report["status"] == "EXPERIMENT_2_COMPLETE"
