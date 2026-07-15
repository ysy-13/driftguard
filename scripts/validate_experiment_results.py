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
    is_phase10 = (args.directory / "manifest.json").exists() and "phase10" in args.directory.as_posix()
    manifest_schema = json.loads((ROOT / f"benchmark/schemas/{'phase10_manifest_schema_v1.json' if is_phase10 else 'experiment_manifest_schema_v1.json'}").read_text())
    record_schema = json.loads((ROOT / "benchmark/schemas/experiment_record_schema_v1.json").read_text())
    phase10_record_schema = json.loads((ROOT / "benchmark/schemas/phase10_record_schema_v1.json").read_text()) if is_phase10 else None
    manifest = json.loads((args.directory / "manifest.json").read_text())
    Draft202012Validator(manifest_schema).validate(manifest)
    record_paths = sorted(args.directory.glob("**/records/*.json")) if is_phase10 else sorted((args.directory / "records").glob("*.json"))
    records_with_paths = [
        (path, json.loads(path.read_text()))
        for path in record_paths if ".cache" not in path.parts
    ]
    records = [record for _, record in records_with_paths]
    forbidden = ("DRIFTGUARD_LLM_API_KEY", "expected_patch_ref", "source_drift_id", "runtime_contract", "variant_code", "chain_of_thought")
    for record in records:
        if is_phase10:
            Draft202012Validator(phase10_record_schema).validate(record)
            phase10_fields = {"provider", "configured_model", "actual_model", "seed", "experiment_stage", "publication_status", "api_calls", "estimated_cost_cny", "failure_analysis", "public_family_id", "sanitized_error"}
            Draft202012Validator(record_schema).validate({key: value for key, value in record.items() if key not in phase10_fields})
        else:
            Draft202012Validator(record_schema).validate(record)
        encoded = json.dumps(record)
        if any(term in encoded for term in forbidden):
            raise ValueError("experiment record contains forbidden data")
    summary = json.loads((args.directory / "summary.json").read_text())
    if not is_phase10 and manifest["mock_provider"] and summary["real_llm_experiment_status"] != "NOT RUN":
        raise ValueError("mock output was mislabeled as a real LLM experiment")
    if is_phase10 and summary["full_real_model_experiment_status"] != "NOT RUN":
        raise ValueError("Pilot was mislabeled as a full real-model experiment")
    print("Experiment result validation passed.")
    print(f"Records: {len(records)}")
    if is_phase10:
        canary_records = sum("canary" in path.parts for path, _ in records_with_paths)
        print(f"Canary records (validated separately): {canary_records}")
        print(f"Non-canary records: {len(records) - canary_records}")
    print(f"Real LLM experiment status: {summary.get('real_llm_experiment_status', 'PILOT')}")
    if is_phase10:
        print("Full real-model experiment status: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
