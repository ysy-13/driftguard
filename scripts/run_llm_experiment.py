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

from driftguard.experiments.runner import LLMExperimentHarness


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the Phase 9 reproducible LLM-agent experiment harness.")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--mode", choices=("component", "end_to_end"))
    value.add_argument("--output", type=Path)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--allow-real-api", action="store_true", help="Explicitly authorize configured real-provider requests.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    harness = LLMExperimentHarness(args.config, args.output, args.allow_real_api, args.dry_run)
    summary = harness.run(args.mode, args.resume)
    print("Phase 9 experiment harness completed.")
    print(f"Experiment: {summary['experiment_id']}")
    print(f"Records: {summary['records']}")
    if not summary.get("dry_run"):
        print(f"Created/resumed: {summary['created_records']}/{summary['resumed_records']}")
        print(f"Modes: {', '.join(summary['modes'])}")
        print(f"Methods: {len(summary['methods'])}")
        print(f"Ground-truth/API-key leakage: {summary['ground_truth_leakage_count']}/{summary['api_key_leakage_count']}")
    print(f"Real LLM experiment status: {summary['real_llm_experiment_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

