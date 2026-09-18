from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from driftguard.specdriftbench.heldout import HeldoutEvidenceViewBuilder
from driftguard.specdriftbench.rule_baseline import ProtocolRuleBaseline


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_specdriftbench_rule_baseline.py"
SPEC = importlib.util.spec_from_file_location("specdriftbench_rule_baseline_runner", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture(scope="module")
def report():
    return RUNNER.build_report()


def test_rule_baseline_uses_all_development_and_heldout_inputs(report):
    assert report["development"]["unique_evidence_records"] == 4 * 3 * 3
    assert report["heldout"]["unique_evidence_records"] == 16 * 3 * 3
    assert set(report["development"]["families"]).isdisjoint(report["heldout"]["families"])


def test_classifier_accepts_only_evidence_and_has_no_family_specific_constants():
    source = (ROOT / "src/driftguard/specdriftbench/rule_baseline.py").read_text(encoding="utf-8")
    assert "family_id" not in source
    assert "variant_code" not in source
    assert "provider_profile" not in source
    assert not any(f'M{index:02d}' in source for index in range(1, 21))


def test_hidden_evaluator_fields_do_not_change_prediction():
    evidence = HeldoutEvidenceViewBuilder().build("M02", "TF", "RETRY_HISTORY")
    classifier = ProtocolRuleBaseline()
    original = classifier.predict(evidence)
    contaminated = copy.deepcopy(evidence)
    contaminated["evaluator_metadata"] = {"ground_truth_label": "persistent_drift"}
    assert classifier.predict(contaminated) == original


def test_rule_order_has_declared_first_failure_fallback(report):
    heldout = report["heldout"]
    assert heldout["fallback_count"] == 16 * 2
    assert heldout["reason_counts"]["FIRST_FAILURE_CONSERVATIVE_PD_FALLBACK"] == 32
    assert all(
        row["evidence_view"] == "FIRST_FAILURE"
        for row in heldout["predictions"]
        if row["used_fallback"]
    )


def test_heldout_rule_metrics_are_internally_consistent(report):
    metric = report["heldout"]["overall"]
    assert metric["records"] == 144
    assert metric["correct"] == 128
    assert metric["accuracy"] == pytest.approx(8 / 9)
    assert metric["macro_f1"] == pytest.approx((1 + 0.8 + 6 / 7) / 3)
    assert sum(sum(row.values()) for row in metric["confusion_matrix"].values()) == 144


def test_view_metrics_show_matched_first_failure_limit(report):
    views = report["heldout"]["by_evidence_view"]
    assert views["FIRST_FAILURE"]["accuracy"] == pytest.approx(2 / 3)
    assert views["FIRST_FAILURE"]["macro_f1"] == pytest.approx(5 / 9)
    assert views["RETRY_HISTORY"]["accuracy"] == 1.0
    assert views["FULL_EVIDENCE"]["accuracy"] == 1.0


def test_provider_alignment_changes_counts_not_rates(report):
    unique = report["heldout"]["overall"]
    aligned = report["heldout_provider_aligned"]
    assert aligned["unique_evidence_records"] == 144
    assert aligned["aligned_rows"] == aligned["records"] == 432
    assert aligned["correct"] == unique["correct"] * 3
    assert aligned["accuracy"] == unique["accuracy"]
    assert aligned["macro_f1"] == unique["macro_f1"]


def test_runner_is_offline_and_does_not_read_model_predictions(report):
    assert report["safety"] == {
        "network_calls": 0,
        "provider_instances": 0,
        "credentials_read": 0,
        "model_predictions_read": 0,
        "ground_truth_available_to_classifier": False,
        "evaluator_labels_used_only_after_prediction": True,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("import requests", "import httpx", "os.environ", "normalized_prediction", "parsed_result")
    assert not any(token in source for token in forbidden)


def test_outputs_are_byte_identical(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    RUNNER.run(first)
    RUNNER.run(second)
    for name in ("protocol_rule_baseline_v1.json", "protocol_rule_baseline_v1.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        if name.endswith(".json"):
            assert json.loads((first / name).read_text(encoding="utf-8"))["schema_version"].endswith("v1")
