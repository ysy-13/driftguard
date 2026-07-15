#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.experiments.metrics import aggregate_metrics
from driftguard.phase10.statistics import holm_correction, mcnemar_exact, paired_family_bootstrap


def load_records(directory: Path):
    records = []
    for path in directory.glob("**/records/*.json"):
        if (
            ".cache" not in path.parts
            and "canary" not in path.parts
            and "symbolic_upper_bound" not in path.parts
        ):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze Phase 10 paired model experiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args(argv)
    records = load_records(args.input)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["provider"], record["mode"], record["method"])].append(record)
    result = {
        "stage": args.stage.upper(),
        "publication_status": "DEVELOPMENT_ONLY" if args.stage == "pilot" else "CANDIDATE_MAIN",
        "descriptive_only": args.stage == "pilot",
        "records": len(records),
        "metrics_by_model_mode_method": {
            "|".join(key): aggregate_metrics(value) for key, value in sorted(grouped.items())
        },
        "table_previews": _tables(records),
        "formal_comparisons": [],
    }
    if args.stage == "main":
        result["formal_comparisons"] = _comparisons(records, args.bootstrap_iterations)
    # Keep every behavior-affecting Pilot attempt self-contained. In
    # particular, pilot_v3 analysis must never overwrite pilot or pilot_v2.
    output = args.input / "analysis" / f"{args.stage}_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Phase 10 {args.stage} analysis completed.")
    print(f"Records: {len(records)}")
    print(f"Descriptive only: {result['descriptive_only']}")
    return 0


def _comparisons(records, iterations):
    comparisons = []
    raw_p = []
    for provider in sorted({item["provider"] for item in records}):
        model_rows = [item for item in records if item["provider"] == provider and item["mode"] == "end_to_end"]
        driftguard = {key(item): item for item in model_rows if item["method"] == "driftguard_llm"}
        for baseline in ("standard", "retry_only", "reflection", "validation_guided"):
            base = {key(item): item for item in model_rows if item["method"] == baseline}
            common = sorted(set(base) & set(driftguard))
            pairs = [{
                "public_family_id": base[item]["public_family_id"],
                "baseline": base[item]["final_task_success"],
                "driftguard": driftguard[item]["final_task_success"],
            } for item in common]
            test = mcnemar_exact([p["baseline"] for p in pairs], [p["driftguard"] for p in pairs])
            bootstrap = paired_family_bootstrap(
                pairs, lambda pair: float(pair["driftguard"]) - float(pair["baseline"]), iterations,
            )
            row = {"provider": provider, "baseline": baseline, "pairs": len(pairs), "mcnemar": test, "bootstrap_95": bootstrap}
            raw_p.append(float(test["p_value"]))
            comparisons.append(row)
    for row, adjusted in zip(comparisons, holm_correction(raw_p)):
        row["holm_adjusted_p"] = adjusted
    return comparisons


def key(record):
    return record["public_scenario_id"], record["seed"], record["repetition"]


def _tables(records):
    return {
        "table_1_benchmark_statistics": {"records": len(records), "models": len({item["configured_model"] for item in records})},
        "table_2_end_to_end_task_success": "metrics_by_model_mode_method",
        "table_3_attribution_localization": "metrics_by_model_mode_method",
        "table_4_patch_repair_transfer": "metrics_by_model_mode_method",
        "table_5_efficiency_cost": {"estimated_cost_cny": sum(item["estimated_cost_cny"] for item in records)},
        "table_6_ablation": "NOT RUN",
        "table_7_safety_false_patch": "metrics_by_model_mode_method",
        "pilot_preview_only": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
