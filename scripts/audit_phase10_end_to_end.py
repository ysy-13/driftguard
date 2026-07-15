#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.phase10.root_cause_audit import Phase10EndToEndRootCauseAudit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline cache-only Phase 10A end-to-end root-cause audit.")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/experiments/phase10/pilot_v3")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    json_path, md_path, report = Phase10EndToEndRootCauseAudit(args.pilot).write(args.output_dir)
    print("Phase 10A end-to-end root-cause audit completed offline.")
    print(f"Records: {report['records_audited']}")
    print(f"Provider calls/new cost: {report['real_api_calls']}/{report['new_cost_cny']:.1f} CNY")
    print(f"Cache/ledger unchanged: {report['integrity']['cache_unchanged']}/{report['integrity']['ledger_unchanged']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print("Full real-model experiment status: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
