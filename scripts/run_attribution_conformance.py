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

from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run Phase 7 evidence-based attribution conformance.")
    value.add_argument("--family")
    value.add_argument("--output", type=Path, default=ROOT / "results" / "attribution" / "attribution_conformance_v1.json")
    value.add_argument("--max-probes", type=int, default=8)
    value.add_argument("--replay", type=int, default=2)
    value.add_argument("--strict-leakage-check", action=argparse.BooleanOptionalAction, default=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = AttributionConformanceRunner(
        max_probes=args.max_probes, replay=args.replay,
        strict_leakage_check=args.strict_leakage_check,
    ).run(args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    total = summary["scenarios"]
    passed = all((
        summary["classification_correct"] == total,
        summary["macro_f1"] == 1.0,
        summary["target_tool_localization_accuracy"] == 1.0,
        summary["drift_category_localization_accuracy"] == 1.0,
        summary["exact_location_accuracy"] == 1.0,
        summary["false_drift_count"] == 0,
        summary["false_patch_eligibility_count"] == 0,
        summary["unsafe_probe_count"] == 0,
        summary["ground_truth_leakage_count"] == 0,
        summary["deterministic_replay_passed"] == total,
        summary["canonical_hashes_unchanged"],
        summary["oracle_regression_passed"] == 32,
    ))
    print("Attribution conformance validation passed." if passed else "Attribution conformance validation failed.")
    print(f"Scenarios: {total}")
    print(f"Classification: {summary['classification_correct']}/{total}")
    print(f"Macro F1: {summary['macro_f1']:.3f}")
    print(f"PD target localization: {summary['target_tool_localization_accuracy']:.3f}")
    print(f"PD category localization: {summary['drift_category_localization_accuracy']:.3f}")
    print(f"Exact location: {summary['exact_location_accuracy']:.3f}")
    print(f"PD patch eligibility recall: {summary['pd_patch_eligibility_recall']:.3f}")
    print(f"Unsafe probes: {summary['unsafe_probe_count']}")
    print(f"Ground-truth leakage: {summary['ground_truth_leakage_count']}")
    print(f"Deterministic replay: {summary['deterministic_replay_passed']}/{total}")
    print(f"Oracle regression: {summary['oracle_regression_passed']}/32")
    print(f"Canonical hashes unchanged: {'yes' if summary['canonical_hashes_unchanged'] else 'no'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
