from __future__ import annotations

import json

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.offline_repair import write_offline_repair_report


def main() -> int:
    path, report = write_offline_repair_report(
        PROJECT_ROOT / "results" / "experiments" / "phase10" / "offline_repair"
    )
    summary = {
        "report": str(path.relative_to(PROJECT_ROOT)),
        "task_consistency": report["task_consistency"],
        "catalog": report["tool_catalog"],
        "offline_oracle": {key: value for key, value in report["offline_oracle"].items() if key != "details"},
        "mock_feedback": {key: value for key, value in report["mock_feedback"].items() if key != "details"},
        "real_provider_calls": report["real_provider_calls"],
        "new_cost_cny": report["new_cost_cny"],
        "pilot_v4_created_or_run": report["pilot_v4_created_or_run"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all((report["task_consistency"]["passed"], report["tool_catalog"]["passed"], report["offline_oracle"]["evaluator_success"] == 12, report["mock_feedback"]["successful_two_round_repairs"] == 5)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
