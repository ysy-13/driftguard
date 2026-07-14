#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.runners.healing_conformance_runner import HealingConformanceRunner


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run Phase 8 evidence-grounded healing conformance.")
    value.add_argument("--family", help="Run one matched family, for example M01.")
    value.add_argument("--output", type=Path, default=ROOT / "results" / "healing" / "healing_conformance_v1.json")
    value.add_argument("--max-candidates", type=int, default=5)
    value.add_argument("--max-repair-calls", type=int, default=8)
    value.add_argument("--strict-leakage-check", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--replay", type=int, default=2)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = HealingConformanceRunner(
        args.max_candidates, args.max_repair_calls, args.strict_leakage_check, args.replay,
    ).run(args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    pd, total = summary["pd_scenarios"], summary["scenarios"]
    passed = all((
        summary["ae_patch_proposal_count"] == 0, summary["tf_patch_proposal_count"] == 0,
        summary["pd_patch_proposal_count"] == pd, summary["pd_accepted_patches"] == pd,
        summary["patch_correctness"] == pd, summary["immediate_repair_success"] == pd,
        summary["future_task_transfer_success"] == pd, summary["regression_pass"] == pd,
        summary["minimality_pass"] == pd, summary["unsafe_patch_count"] == 0,
        summary["false_patch_count"] == 0, summary["main_state_pollution_count"] == 0,
        summary["ground_truth_leakage_count"] == 0, summary["deterministic_replay_passed"] == total,
        summary["canonical_hashes_unchanged"], summary["canonical_handlers_unchanged"],
        summary["oracle_regression_passed"] == 32,
    ))
    print("Healing conformance validation passed." if passed else "Healing conformance validation failed.")
    print(f"Scenarios: {total}; PD: {pd}")
    print(f"AE/TF patch proposals: {summary['ae_patch_proposal_count']}/{summary['tf_patch_proposal_count']}")
    print(f"PD proposed/accepted/correct: {summary['pd_patch_proposal_count']}/{summary['pd_accepted_patches']}/{summary['patch_correctness']}")
    print(f"Immediate repair: {summary['immediate_repair_success']}/{pd}")
    print(f"Future transfer: {summary['future_task_transfer_success']}/{pd}")
    print(f"Regression/minimality: {summary['regression_pass']}/{summary['minimality_pass']}")
    print(f"Unsafe/false/leakage: {summary['unsafe_patch_count']}/{summary['false_patch_count']}/{summary['ground_truth_leakage_count']}")
    print(f"Deterministic replay: {summary['deterministic_replay_passed']}/{total}")
    print(f"Oracle regression: {summary['oracle_regression_passed']}/32")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

