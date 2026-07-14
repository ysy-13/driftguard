import json

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.runners.healing_conformance_runner import HealingConformanceRunner


def test_single_family_runner_covers_all_three_variants_and_real_repair():
    result = HealingConformanceRunner(replay=2).run("M01")
    summary = result["summary"]
    assert summary["scenarios"] == 3
    assert summary["ae_patch_proposal_count"] == 0
    assert summary["tf_patch_proposal_count"] == 0
    assert summary["pd_patch_proposal_count"] == 1
    assert summary["pd_accepted_patches"] == 1
    assert summary["immediate_repair_success"] == 1
    assert summary["future_task_transfer_success"] == 1
    assert summary["deterministic_replay_passed"] == 3
    assert summary["canonical_hashes_unchanged"]
    assert summary["canonical_handlers_unchanged"]


def test_formal_60_scenario_result_meets_phase8_targets():
    result = json.loads((PROJECT_ROOT / "results/healing/healing_conformance_v1.json").read_text())
    summary = result["summary"]
    assert summary["scenarios"] == 60 and summary["pd_scenarios"] == 20
    assert summary["ae_patch_proposal_count"] == summary["tf_patch_proposal_count"] == 0
    assert summary["pd_patch_proposal_count"] == summary["pd_accepted_patches"] == 20
    assert summary["patch_correctness"] == summary["immediate_repair_success"] == 20
    assert summary["future_task_transfer_success"] == summary["regression_pass"] == 20
    assert summary["minimality_pass"] == 20
    assert summary["unsafe_patch_count"] == summary["false_patch_count"] == 0


def test_held_out_results_are_first_try_and_cannot_rewrite_patch():
    result = json.loads((PROJECT_ROOT / "results/healing/healing_conformance_v1.json").read_text())
    transfers = [row["future_transfer"] for row in result["scenarios"] if row["future_transfer"]]
    assert len(transfers) == 20
    assert all(item["held_out_episode"] == 6 and item["first_result_counted"] for item in transfers)
    assert all(item["same_initial_state"] and not item["candidate_rewritten_after_transfer"] for item in transfers)
    assert all(not item["no_patch_success"] and item["patched_success"] for item in transfers)


def test_formal_result_protects_canonical_files_handlers_and_prior_phases():
    result = json.loads((PROJECT_ROOT / "results/healing/healing_conformance_v1.json").read_text())
    summary = result["summary"]
    assert result["canonical_hashes_before"] == result["canonical_hashes_after"]
    assert result["handler_hashes_before"] == result["handler_hashes_after"]
    assert summary["canonical_hashes_unchanged"] and summary["canonical_handlers_unchanged"]
    assert summary["oracle_regression_passed"] == 32
    assert summary["phase6_regression_passed"] == 20
    assert summary["phase7_regression_passed"] == 60
