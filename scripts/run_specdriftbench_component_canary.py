#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.credentials import load_project_dotenv
from driftguard.specdriftbench.runner import (
    CONFIG_PATH, SpecDriftBenchConfig, SpecDriftBenchRunner,
    offline_rescore_v1, prepare_offline_assets, run_kimi_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SpecDriftBench Component Canary protocol.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true")
    modes.add_argument("--offline-prepare", action="store_true")
    modes.add_argument("--offline-rescore", action="store_true")
    modes.add_argument("--kimi-preflight", action="store_true")
    modes.add_argument("--real", action="store_true")
    modes.add_argument("--replay", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--attempt-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.validate_config:
        config = SpecDriftBenchConfig.load(args.config)
        report = {"passed": True, "records": len(config.plan), "heldout48_overlap": 0,
                  "plan": [item.to_dict() for item in config.plan]}
    elif args.offline_prepare:
        load_project_dotenv(PROJECT_ROOT)
        report = prepare_offline_assets(args.output, args.config)
    elif args.offline_rescore:
        report = offline_rescore_v1(output=args.output)
    else:
        if not args.attempt_id:
            parser.error("--attempt-id is required for preflight/real/replay")
        if args.kimi_preflight or args.real:
            load_project_dotenv(PROJECT_ROOT)
        if args.kimi_preflight:
            report = run_kimi_preflight(
                args.config, attempt_id=args.attempt_id,
                allow_real_api=args.allow_real_api, confirmation=args.confirmation,
                output_root=args.output,
            )
        else:
            runner = SpecDriftBenchRunner(
                args.config, attempt_id=args.attempt_id,
                mode="real" if args.real else "replay",
                allow_real_api=args.allow_real_api, confirmation=args.confirmation,
                output_root=args.output,
            )
            report = runner.run(resume=args.resume)
    printable = report.get("summary", report)
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
