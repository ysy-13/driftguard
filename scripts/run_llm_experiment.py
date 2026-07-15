#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.experiments.runner import LLMExperimentHarness
from driftguard.phase10.runner import Phase10Execution


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the Phase 9 reproducible LLM-agent experiment harness.")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--mode", choices=("component", "end_to_end"))
    value.add_argument("--output", type=Path)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--allow-real-api", action="store_true", help="Explicitly authorize configured real-provider requests.")
    value.add_argument("--preflight", action="store_true", help="Run only the Phase 10 real-provider preflight.")
    value.add_argument("--confirm-full-run", action="store_true", help="Separate confirmation gate for Phase 10B/10C configs.")
    value.add_argument("--component-canary", choices=("deepseek", "dashscope"), help="Run only the selected Phase 10A three-record component canary.")
    value.add_argument("--end-to-end-canary", choices=("deepseek", "dashscope"), help="Run only the selected Phase 10A five-record end-to-end canary.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "experiment" in raw and "models" in raw:
        execution = Phase10Execution(
            args.config, args.output, args.allow_real_api, args.confirm_full_run,
        )
        for line in execution.safe_credential_lines():
            print(line)
        if args.preflight:
            summary = execution.run_preflight()
            print("Phase 10A provider preflight completed.")
            print(f"Passed models: {len(summary['passed_models'])}/{len(execution.config.models)}")
            print(f"Estimated cumulative cost (CNY): {summary['cost']['spent_cny']:.6f}")
            print("Full real-model experiment status: NOT RUN")
            return 0 if len(summary["passed_models"]) == len(execution.config.models) else 2
        if args.component_canary:
            report = execution.run_component_canary(args.component_canary)
            print("Phase 10A component canary completed.")
            print(f"Provider: {report['provider']}")
            print(f"Records/errors: {report['records']}/{report['infrastructure_errors']}")
            print(f"Infrastructure error rate: {report['infrastructure_error_rate']:.3f}")
            print("Full real-model experiment status: NOT RUN")
            return 0 if report["infrastructure_errors"] == 0 else 2
        if args.end_to_end_canary:
            report = execution.run_end_to_end_canary(args.end_to_end_canary)
            print("Phase 10A end-to-end canary completed.")
            print(f"Provider: {report['provider']}")
            print(f"Records/schema-valid: {report['records']}/{report['structured_valid_records']}")
            print(f"Parsed TOOL_CALL records: {report['parsed_tool_call_records']}")
            print(f"Infrastructure error rate: {report['infrastructure_error_rate']:.3f}")
            print("Full real-model experiment status: NOT RUN")
            passed = (
                report["records"] == 5
                and report["structured_valid_records"] == 5
                and report["infrastructure_errors"] == 0
                and report["tool_call_interface_verified"]
            )
            return 0 if passed else 2
        summary = execution.run_pilot(args.resume)
        print("Phase 10A real-model Pilot completed.")
        print(f"Real model records: {summary['real_model_records']}/{summary['planned_real_model_records']}")
        print(f"Estimated cumulative cost (CNY): {summary['cost']['spent_cny']:.6f}")
        print(f"API-key/ground-truth leakage: {summary['api_key_leakage_count']}/{summary['ground_truth_leakage_count']}")
        print("Full real-model experiment status: NOT RUN")
        return 0
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
