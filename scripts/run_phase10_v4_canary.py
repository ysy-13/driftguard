from __future__ import annotations

import argparse
import json

from driftguard.phase10.v4_canary import Phase10V4Canary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the isolated 10-record Phase 10A Pilot v4 Canary.")
    value.add_argument("--preflight", action="store_true", help="Run offline checks only; never call a Provider.")
    value.add_argument("--execute", action="store_true", help="Run the authorized DeepSeek then Qwen Canary.")
    value.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.preflight == args.execute:
        raise SystemExit("select exactly one of --preflight or --execute")
    canary = Phase10V4Canary(allow_real_api=args.execute)
    if args.preflight:
        report = canary.preflight(write=True)
        print(json.dumps({
            "passed": report["passed"], "checks": report["checks"],
            "credentials": report["credentials"], "cost_estimate": report["cost_estimate"],
            "real_provider_calls": 0, "full_real_model_experiment_status": "NOT RUN",
        }, indent=2, sort_keys=True))
        return 0
    report = canary.run(resume=args.resume)
    print(json.dumps({
        "records": report["records"], "provider_reports": report["provider_reports"],
        "task_success": report["task_success"], "infrastructure_errors": report["infrastructure_errors"],
        "cost": report["cost"], "cache_replay": report["cache_replay"],
        "full_real_model_experiment_status": "NOT RUN",
    }, indent=2, sort_keys=True))
    return 0 if report["records"] == 10 and not report["halted"] and report["cache_replay"]["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
