#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.credentials import load_project_dotenv
from driftguard.specdriftbench.heldout import (
    CONFIG_PATH,
    HeldoutRunner,
    run_offline_fake_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, replay, or explicitly authorize the frozen SpecDriftBench held-out Component protocol."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true")
    modes.add_argument("--fake-validate", action="store_true")
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
        if args.allow_real_api or args.confirmation is not None or args.resume:
            parser.error("offline config validation rejects real-API and resume flags")
        report = HeldoutRunner.validate_config(args.config)
    elif args.fake_validate:
        if not args.output:
            parser.error("--output is required for FakeProvider validation")
        if args.allow_real_api or args.confirmation is not None or args.resume:
            parser.error("FakeProvider validation rejects real-API and resume flags")
        report = run_offline_fake_validation(
            args.output, attempt_id=args.attempt_id or "heldout432-fake-validation-v1",
            config_path=args.config,
        )
    else:
        if not args.attempt_id:
            parser.error("--attempt-id is required for real or replay mode")
        if args.real:
            # Credentials are loaded only after the caller explicitly selects real mode.
            load_project_dotenv(PROJECT_ROOT)
        runner = HeldoutRunner(
            args.config, attempt_id=args.attempt_id,
            mode="real" if args.real else "replay",
            output_root=args.output, allow_real_api=args.allow_real_api,
            confirmation=args.confirmation,
        )
        report = runner.run(resume=args.resume)

    print(json.dumps(report.get("summary", report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
