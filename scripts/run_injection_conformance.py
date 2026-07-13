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

from driftguard.runners.injection_conformance_runner import InjectionConformanceRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 6 first-failure injection conformance checks.")
    parser.add_argument("--family", help="Run one matched family, for example M01.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "injection" / "injection_conformance_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = InjectionConformanceRunner().run(args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    total = summary["families"]
    passed = all(
        (
            summary["agent_error_executed"] == total,
            summary["agent_error_corrected"] == total,
            summary["transient_failure_executed"] == total,
            summary["persistent_drift_executed"] == total,
            summary["shared_signature_matched"] == total,
            summary["transient_recovered"] == total,
            summary["persistent_reproduced"] == total,
            summary["canonical_hash_unchanged"],
            summary["oracle_regression_passed"] == 32,
        )
    )
    print("Injection conformance validation passed." if passed else "Injection conformance validation failed.")
    print(f"Families: {total}")
    print(f"First-failure scenarios: {summary['first_failure_scenarios']}")
    print(f"Agent Error executed: {summary['agent_error_executed']}/{total}")
    print(f"Transient Failure executed: {summary['transient_failure_executed']}/{total}")
    print(f"Persistent Drift executed: {summary['persistent_drift_executed']}/{total}")
    print(f"Shared signatures matched: {summary['shared_signature_matched']}/{total}")
    print(f"Transient recovery: {summary['transient_recovered']}/{total}")
    print(f"Persistent reproduction: {summary['persistent_reproduced']}/{total}")
    print(f"Canonical hashes unchanged: {'yes' if summary['canonical_hash_unchanged'] else 'no'}")
    print(f"Oracle regression: {summary['oracle_regression_passed']}/32")
    if not passed:
        for family in report["families"]:
            for variant, details in family["variant_details"].items():
                if not details["first_failure_matched"]:
                    print(f"{family['matched_case_id']} {variant}")
                    print(f"expected: shared family signature")
                    print(f"actual: {json.dumps(details['normalized_signature'], ensure_ascii=False)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
