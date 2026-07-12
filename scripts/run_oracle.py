#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from driftguard.runners import OracleRunner
from driftguard.runners.result_writer import write_results


DEFAULT_OUTPUT = Path("results/oracle/oracle_results_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DriftGuard canonical oracle tasks.")
    parser.add_argument("--task", help="Run one canonical task ID.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = OracleRunner().run_all(replay=args.replay, task_id=args.task)
    except (ValueError, OSError) as exc:
        print(f"Oracle sandbox validation failed: {exc}")
        return 1
    write_results(results, args.output)
    summary = results["summary"]
    if summary["failed"]:
        print("Oracle sandbox validation failed.")
        for task in results["tasks"]:
            if not task["task_success"]:
                print(f"{task['task_id']}: {task['failure_details']}")
        return 1
    print("Oracle sandbox validation passed.")
    print(f"Tasks: {summary['tasks']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Forbidden side effects: {summary['forbidden_side_effects']}")
    print(
        "Deterministic replay: "
        f"{summary['deterministic_replay_passed']}/{summary['tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
