from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import yaml

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.contracts.loader import load_openapi
from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.llm import ModelConfig


FOCUSED_CONFIG = PROJECT_ROOT / "configs/experiments/phase10_driftguard_focused_canary.yaml"
EXPERIMENT_KIND = "FOCUSED_LIVE_HEALING_CANARY"
EXPECTED_SELECTION = (("M01", "M01-PD"), ("M06", "M06-PD"), ("M11", "M11-PD"), ("M16", "M16-PD"))


@dataclass(frozen=True)
class FocusedRecordPlan:
    ordinal: int
    provider: str
    model_id: str
    family: str
    scenario_id: str
    method: str
    repetition: int
    seed: int

    @property
    def identity(self) -> str:
        material = f"{EXPERIMENT_KIND}|{self.provider}|{self.model_id}|{self.family}|{self.scenario_id}|{self.method}|{self.repetition}|{self.seed}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal, "record_id": self.identity,
            "provider": self.provider, "model_id": self.model_id,
            "family": self.family, "scenario_id": self.scenario_id,
            "method": self.method, "repetition": self.repetition, "seed": self.seed,
        }


@dataclass(frozen=True)
class FocusedLiveHealingConfig:
    path: Path
    raw: dict[str, Any]
    models: tuple[ModelConfig, ...]
    plan: tuple[FocusedRecordPlan, ...]
    config_hash: str

    @classmethod
    def load(cls, path: Path | str = FOCUSED_CONFIG) -> "FocusedLiveHealingConfig":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Focused config root must be a mapping")
        report = validate_focused_canary_config(source)
        if not report["passed"]:
            failed = sorted(key for key, passed in report["checks"].items() if not passed)
            raise ValueError("Focused config validation failed: " + ", ".join(failed))
        models = tuple(ModelConfig.from_mapping(item) for item in raw["models"])
        seed = int(raw["experiment"]["seeds"][0])
        selection = tuple((item["family"], item["scenario_id"]) for item in raw["experiment"]["selection"])
        plan = tuple(
            FocusedRecordPlan(
                ordinal=index + 1, provider=model.provider, model_id=model.model_id,
                family=family, scenario_id=scenario, method="driftguard_llm",
                repetition=0, seed=seed,
            )
            for index, (model, (family, scenario)) in enumerate(
                ((model, pair) for model in models for pair in selection)
            )
        )
        if len(plan) != 8:
            raise ValueError(f"Focused plan must contain exactly 8 records, got {len(plan)}")
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        return cls(source, raw, models, plan, hashlib.sha256(encoded.encode()).hexdigest())

    @property
    def run_authorized(self) -> bool:
        return bool(self.raw["experiment"]["run_authorized"])

    @property
    def experiment_kind(self) -> str:
        return EXPERIMENT_KIND

    @property
    def output_directory(self) -> Path:
        return PROJECT_ROOT / self.raw["experiment"]["output_directory"]

    def plan_dicts(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.plan]


def validate_focused_canary_config(path: Path | str = FOCUSED_CONFIG) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    experiment, execution, budget = raw["experiment"], raw["execution"], raw["budget"]
    selection = experiment["selection"]
    providers = {(item["provider"], item["model_id"]) for item in raw["models"]}
    expected_scenarios = {"M01-PD", "M06-PD", "M11-PD", "M16-PD"}
    fingerprints = raw["fingerprints"]
    expected_hashes = {
        "action_prompt_v3_sha256": "benchmark/prompts/base_tool_agent_v3.txt",
        "action_schema_v3_sha256": "benchmark/schemas/agent_action_schema_v3.json",
        "attribution_prompt_v3_sha256": "benchmark/prompts/component_attribution_v3.txt",
        "attribution_schema_v3_sha256": "benchmark/schemas/llm_attribution_schema_v3.json",
        "probe_prompt_v3_sha256": "benchmark/prompts/live/driftguard_probe_selection_v3.txt",
        "probe_schema_v3_sha256": "benchmark/schemas/probe_selection_schema_v3.json",
        "patch_prompt_v3_sha256": "benchmark/prompts/live/driftguard_patch_v3.txt",
        "patch_schema_v3_sha256": "benchmark/schemas/llm_patch_proposal_schema_v3.json",
    }
    hashes_valid = all(
        fingerprints[name] == hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for name, relative in expected_hashes.items()
    )
    catalog_valid = fingerprints["tool_catalog_fingerprint"] == ToolCatalogRenderer().fingerprint(load_openapi())
    checks = {
        "experiment_kind": experiment.get("kind") == EXPERIMENT_KIND,
        "authorization_flag_is_boolean": isinstance(experiment.get("run_authorized"), bool),
        "providers": providers == {("deepseek", "deepseek-v4-flash"), ("dashscope", "qwen3.7-plus")},
        "provider_order": [item["provider"] for item in raw["models"]] == ["deepseek", "dashscope"],
        "method": raw["methods"]["end_to_end"] == ["driftguard_llm"] and not raw["methods"]["component"],
        "development_pd_only": {item["scenario_id"] for item in selection} == expected_scenarios
            and all(item["variant"] == "PD" for item in selection),
        "records": len(raw["models"]) * len(selection) * len(raw["methods"]["end_to_end"])
            * experiment["repetitions"] == experiment["planned_records"] == 8,
        "selection_order": tuple((item["family"], item["scenario_id"]) for item in selection) == EXPECTED_SELECTION,
        "repetition": experiment["repetitions"] == 1 and experiment["seeds"] == [20260715],
        "heldout48_overlap": not experiment["heldout48_allowed"] and not any(
            item["family"] not in {"M01", "M06", "M11", "M16"} for item in selection
        ),
        "isolated_namespaces": execution["cache_namespace"] == "driftguard_focused_live_healing_v3"
            and execution["result_namespace"] == "driftguard_focused_canary_v3",
        "ledger_continuity": execution["attempt_base_spent_cny"] == 5.202644120,
        "cost_gates": execution["attempt_soft_increment_cny"] == 3
            and execution["attempt_hard_increment_cny"] == 5
            and budget["full_hard_limit_cny"] == 50,
        "future_acceptance_gate": raw["acceptance"]["stop_if_provider_patch_proposals_zero"] is True,
        "frozen_fingerprints": hashes_valid and catalog_valid,
        "cache_identity_complete": set(fingerprints["cache_key_material"]) == {
            "action_prompt_hash", "action_schema_hash", "attribution_prompt_hash", "attribution_schema_hash",
            "probe_prompt_hash", "probe_schema_hash", "patch_prompt_hash", "patch_schema_hash",
            "tool_catalog_fingerprint", "policy_capabilities", "provider", "model", "method",
            "public_scenario_id", "repetition", "attempt_version", "episode_stage",
            "evidence_fingerprint", "source_snapshot", "config_hash",
        },
    }
    return {"passed": all(checks.values()), "checks": checks, "planned_records": 8, "heldout48_overlap": 0}
