#!/usr/bin/env python3
"""Run the offline protocol-rule baseline on development and held-out views."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from driftguard.specdriftbench.heldout import (
    EVIDENCE_VIEWS,
    HELDOUT_FAMILIES,
    VARIANTS,
    HeldoutEvidenceViewBuilder,
    assert_heldout_provider_visible,
)
from driftguard.specdriftbench.protocol import (
    FAMILIES as DEVELOPMENT_FAMILIES,
    EvidenceViewBuilder,
    expected_for_evaluator,
)
from driftguard.specdriftbench.rule_baseline import (
    CLASSES,
    ProtocolRuleBaseline,
    classification_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ID = "specdriftbench-heldout432-v2-20260719-3f4fd32-01"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results/experiments/phase11/specdriftbench_component_heldout432/attempts"
    / ATTEMPT_ID
    / "analysis/rule_baseline"
)
VARIANT_CLASS = {
    "AE": "AGENT_ERROR",
    "TF": "TRANSIENT_FAILURE",
    "PD": "PERSISTENT_DRIFT",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_split(
    builder: EvidenceViewBuilder,
    families: tuple[str, ...],
) -> dict[str, Any]:
    classifier = ProtocolRuleBaseline()
    rows: list[dict[str, Any]] = []
    for family in families:
        for variant in VARIANTS:
            expected = expected_for_evaluator(builder, family, variant)["predicted_class"]
            if expected != VARIANT_CLASS[variant]:
                raise ValueError("evaluator class mapping changed")
            for view in EVIDENCE_VIEWS:
                evidence = builder.build(family, variant, view)
                assert_heldout_provider_visible(evidence)
                prediction = classifier.predict(evidence)
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "evidence_view": view,
                        "public_scenario_id": evidence["public_scenario_id"],
                        "evidence_sha256": evidence["evidence_sha256"],
                        "actual_class": expected,
                        **prediction.to_dict(),
                        "correct": prediction.predicted_class == expected,
                    }
                )

    overall = classification_metrics(
        [(row["actual_class"], row["predicted_class"]) for row in rows]
    )
    by_view = {
        view: classification_metrics(
            [
                (row["actual_class"], row["predicted_class"])
                for row in rows
                if row["evidence_view"] == view
            ]
        )
        for view in EVIDENCE_VIEWS
    }
    reason_counts = Counter(row["reason_code"] for row in rows)
    return {
        "families": list(families),
        "unique_evidence_records": len(rows),
        "overall": overall,
        "by_evidence_view": by_view,
        "fallback_count": sum(row["used_fallback"] for row in rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "errors": [row for row in rows if not row["correct"]],
        "predictions": rows,
    }


def provider_aligned_metrics(heldout: dict[str, Any]) -> dict[str, Any]:
    """Replicate deterministic predictions only for a 432-row metric comparison.

    There are still only 144 unique evidence inputs.  Replication across the
    three provider slots changes counts, not rates, and is never treated as
    additional independent evidence.
    """
    pairs = [
        (row["actual_class"], row["predicted_class"])
        for row in heldout["predictions"]
        for _ in range(3)
    ]
    return {
        "provider_slots": 3,
        "unique_evidence_records": heldout["unique_evidence_records"],
        "aligned_rows": len(pairs),
        "independence_note": "The 432 rows repeat 144 deterministic predictions across three provider slots; rates are unchanged and inferential sample size is not increased.",
        **classification_metrics(pairs),
    }


def build_report() -> dict[str, Any]:
    development = evaluate_split(EvidenceViewBuilder(), tuple(DEVELOPMENT_FAMILIES))
    heldout = evaluate_split(HeldoutEvidenceViewBuilder(), tuple(HELDOUT_FAMILIES))
    return {
        "schema_version": "specdriftbench-protocol-rule-baseline-v1",
        "status": "POST_HOC_CAMERA_READY_BASELINE",
        "method": {
            "name": "Protocol-Rule",
            "input_boundary": "the same Agent-visible evidence object supplied to the LLMs",
            "rules_in_order": [
                "visible call or response interpretation violates the displayed contract -> AGENT_ERROR",
                "exact legal retry recovers -> TRANSIENT_FAILURE",
                "exact legal retry reproduces the failure -> PERSISTENT_DRIFT",
                "otherwise, a conforming first failure uses a declared conservative PERSISTENT_DRIFT fallback",
            ],
            "family_specific_rules": 0,
            "provider_specific_rules": 0,
            "learned_parameters": 0,
            "api_calls": 0,
            "model_calls": 0,
        },
        "development": development,
        "heldout": heldout,
        "heldout_provider_aligned": provider_aligned_metrics(heldout),
        "safety": {
            "network_calls": 0,
            "provider_instances": 0,
            "credentials_read": 0,
            "model_predictions_read": 0,
            "ground_truth_available_to_classifier": False,
            "evaluator_labels_used_only_after_prediction": True,
        },
        "source": {
            "classifier": "src/driftguard/specdriftbench/rule_baseline.py",
            "runner": "scripts/run_specdriftbench_rule_baseline.py",
            "classifier_sha256": sha256_file(
                PROJECT_ROOT / "src/driftguard/specdriftbench/rule_baseline.py"
            ),
        },
        "limitations": [
            "This baseline was added after peer review and was not preregistered with the original analysis plan.",
            "The conservative first-failure fallback is a declared tie-breaking policy, not evidence that drift is present.",
            "The rule baseline evaluates causal classification only; it does not provide drift-category or location predictions.",
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    heldout = report["heldout"]
    lines = [
        "# SpecDriftBench Protocol-Rule Baseline",
        "",
        "Status: `POST_HOC_CAMERA_READY_BASELINE`.",
        "",
        "The classifier consumes only the same Agent-visible evidence supplied to the LLMs. It makes no model or network calls and has no family- or provider-specific rules.",
        "",
        "## Rules",
        "",
    ]
    lines.extend(
        f"{index}. {rule}"
        for index, rule in enumerate(report["method"]["rules_in_order"], 1)
    )
    lines += [
        "",
        "## Held-out results (144 unique evidence inputs)",
        "",
        "| View | Accuracy | Macro-F1 | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for view in EVIDENCE_VIEWS:
        metric = heldout["by_evidence_view"][view]
        lines.append(
            f"| {view} | {fmt(metric['accuracy'])} | {fmt(metric['macro_f1'])} | {fmt(metric['coverage'])} |"
        )
    overall = heldout["overall"]
    lines += [
        f"| Overall | {fmt(overall['accuracy'])} | {fmt(overall['macro_f1'])} | {fmt(overall['coverage'])} |",
        "",
        "The provider-aligned 432-row expansion repeats each deterministic prediction in the three provider slots. It has identical rates and does not create additional independent observations.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    report = build_report()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "protocol_rule_baseline_v1.json"
    md_path = output_root / "protocol_rule_baseline_v1.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = run(args.output_root)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "heldout_unique_records": report["heldout"]["unique_evidence_records"],
                "heldout_accuracy": report["heldout"]["overall"]["accuracy"],
                "heldout_macro_f1": report["heldout"]["overall"]["macro_f1"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
