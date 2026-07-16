#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.runners.live_healing_mock_runner import LiveHealingMockRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fully offline DriftGuard live-healing Mock chains.")
    parser.add_argument("--output", type=Path, default=Path("results/experiments/phase10/live_healing_mock_v1.json"))
    args = parser.parse_args()
    report = LiveHealingMockRunner().run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["accepted"] == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
