#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.live.focused_config import FOCUSED_CONFIG
from driftguard.phase10.credentials import load_project_dotenv
from driftguard.runners.focused_live_healing_runner import FocusedLiveHealingRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dedicated Focused Live-Healing Canary.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--replay", action="store_true")
    modes.add_argument("--real", action="store_true")
    modes.add_argument("--replay-real", action="store_true")
    parser.add_argument("--config", type=Path, default=FOCUSED_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt-id")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--confirm-focused-canary")
    args = parser.parse_args()

    if args.validate_config:
        report = FocusedLiveHealingRunner.validate_config(args.config)
    elif args.dry_run:
        report = FocusedLiveHealingRunner.dry_run(args.config)
    else:
        mode = (
            "mock" if args.mock else "replay" if args.replay
            else "real" if args.real else "replay-real"
        )
        if mode == "real":
            load_project_dotenv(PROJECT_ROOT)
        default_output = None
        if mode in {"mock", "replay"}:
            default_output = PROJECT_ROOT / "results/experiments/phase10/driftguard_focused_canary" / mode
        runner = FocusedLiveHealingRunner(
            args.config, mode=mode, output=args.output or default_output,
            cache_root=args.cache_root,
            attempt_id=args.attempt_id,
            allow_real_api=args.allow_real_api,
            confirm_focused_canary=args.confirm_focused_canary,
        )
        report = runner.run(resume=args.resume)
    printable = report["summary"] if "summary" in report else report
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
