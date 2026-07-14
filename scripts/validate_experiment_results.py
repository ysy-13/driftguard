#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 9 experiment manifests and records.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    manifest_schema = json.loads((ROOT / "benchmark/schemas/experiment_manifest_schema_v1.json").read_text())
    record_schema = json.loads((ROOT / "benchmark/schemas/experiment_record_schema_v1.json").read_text())
    manifest = json.loads((args.directory / "manifest.json").read_text())
    Draft202012Validator(manifest_schema).validate(manifest)
    records = [json.loads(path.read_text()) for path in sorted((args.directory / "records").glob("*.json"))]
    forbidden = ("DRIFTGUARD_LLM_API_KEY", "expected_patch_ref", "source_drift_id", "runtime_contract", "variant_code", "chain_of_thought")
    for record in records:
        Draft202012Validator(record_schema).validate(record)
        encoded = json.dumps(record)
        if any(term in encoded for term in forbidden):
            raise ValueError("experiment record contains forbidden data")
    summary = json.loads((args.directory / "summary.json").read_text())
    if manifest["mock_provider"] and summary["real_llm_experiment_status"] != "NOT RUN":
        raise ValueError("mock output was mislabeled as a real LLM experiment")
    print("Experiment result validation passed.")
    print(f"Records: {len(records)}")
    print(f"Real LLM experiment status: {summary['real_llm_experiment_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
