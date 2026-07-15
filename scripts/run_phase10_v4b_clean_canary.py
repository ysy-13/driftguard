#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.consistency_audit import source_snapshot
from driftguard.phase10.v4_canary import Phase10V4Canary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated Phase 10A v4b clean canary.")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    runner = Phase10V4Canary(
        PROJECT_ROOT / "configs/experiments/phase10_real_pilot_v4b_clean_canary.yaml",
        PROJECT_ROOT / "results/experiments/phase10/pilot_v4b_clean_canary",
        args.allow_real_api,
        attempt="v4b_clean_canary", cache_namespace="v4b_clean_canary",
        base_spent_cny=4.396109052, incremental_soft_cny=1.0,
        incremental_hard_cny=2.0, total_hard_cny=50.0,
        frozen_source_snapshot=source_snapshot(),
    )
    result = runner.preflight() if args.preflight else runner.run(resume=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
