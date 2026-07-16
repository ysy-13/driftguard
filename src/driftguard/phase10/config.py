from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from driftguard.experiments.budgets import ExperimentBudget
from driftguard.experiments.config import ExperimentConfig
from driftguard.llm import ModelCapabilities, ModelConfig


FIXED_MODELS = {
    "deepseek": ("https://api.deepseek.com", "deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.7-plus", "DASHSCOPE_API_KEY"),
}


@dataclass(frozen=True)
class Phase10Budget:
    pilot_hard_limit_cny: float
    pilot_soft_limit_cny: float
    full_hard_limit_cny: float
    full_soft_limit_cny: float
    max_llm_calls_per_record: int
    max_tool_calls_per_record: int
    max_probe_calls_per_record: int
    max_total_interactions_per_record: int
    max_input_tokens_per_record: int
    max_output_tokens_per_record: int
    max_format_repairs: int
    reserve_worst_case_before_call: bool
    max_patch_proposal_calls_per_record: int = 2


@dataclass(frozen=True)
class Phase10Config:
    name: str
    modes: tuple[str, ...]
    real_llm: bool
    repetitions: int
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    models: tuple[ModelConfig, ...]
    methods: dict[str, tuple[str, ...]]
    budget: Phase10Budget
    execution: dict[str, Any]
    stage: str
    raw: dict[str, Any]
    config_hash: str

    @classmethod
    def load(cls, path: Path | str) -> "Phase10Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "experiment" not in raw or "models" not in raw:
            raise ValueError("not a Phase 10 experiment config")
        if raw["experiment"].get("kind") == "FOCUSED_LIVE_HEALING_CANARY":
            raise ValueError(
                "Focused Live-Healing configs must use FocusedLiveHealingConfig; "
                "the generic Phase10Config MAIN path is forbidden"
            )
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        experiment = raw["experiment"]
        models = tuple(ModelConfig.from_mapping(item) for item in raw["models"])
        budget = Phase10Budget(**raw["budget"])
        methods = {key: tuple(value) for key, value in raw["methods"].items()}
        name = str(experiment["name"])
        stage = "PILOT" if "pilot" in name else "ABLATION" if "ablation" in name else "MAIN"
        result = cls(
            name, tuple(experiment["mode"]), bool(experiment["real_llm"]),
            int(experiment["repetitions"]), tuple(int(item) for item in experiment["seeds"]),
            tuple(experiment["families"]), models, methods, budget,
            dict(raw["execution"]), stage, raw,
            hashlib.sha256(encoded.encode()).hexdigest(),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.repetitions != len(self.seeds):
            raise ValueError("each repetition requires exactly one seed")
        if len(set((model.provider, model.model_id) for model in self.models)) != len(self.models):
            raise ValueError("duplicate model configuration")
        for model in self.models:
            if model.provider not in FIXED_MODELS:
                raise ValueError(f"Phase 10 provider is fixed: {model.provider}")
            expected = FIXED_MODELS[model.provider]
            if (model.base_url, model.model_id, model.api_key_env) != expected:
                raise ValueError(f"fixed Phase 10 model mismatch for {model.provider}")
            if model.temperature != 0.1 or model.thinking_mode != "disabled":
                raise ValueError("Phase 10 requires temperature=0.1 and thinking disabled")
        if self.stage == "PILOT" and (self.repetitions != 1 or set(self.families) != {"M01", "M06", "M11", "M16"}):
            raise ValueError("Pilot selection is fixed to four families and one repetition")

    def phase9_view(self, model: ModelConfig, mode: str, methods: tuple[str, ...], seed: int) -> ExperimentConfig:
        per_record = ExperimentBudget(
            max_llm_calls=self.budget.max_llm_calls_per_record,
            max_tool_calls=self.budget.max_tool_calls_per_record,
            max_probe_calls=self.budget.max_probe_calls_per_record,
            max_total_interactions=self.budget.max_total_interactions_per_record,
            max_input_tokens=self.budget.max_input_tokens_per_record,
            max_output_tokens=self.budget.max_output_tokens_per_record,
            max_wall_time_seconds=float(self.execution.get("max_wall_time_seconds", 300)),
            max_format_repairs=self.budget.max_format_repairs,
            max_patch_proposal_calls=self.budget.max_patch_proposal_calls_per_record,
        )
        seeded = ModelConfig(**{**model.__dict__, "seed": seed})
        return ExperimentConfig(
            self.name, seeded, self.families, methods, (mode,), 1, (seed,),
            int(self.execution.get("parallelism", model.concurrency)),
            float(self.execution["rate_limit_per_second"]) if self.execution.get("rate_limit_per_second") else None,
            int(self.execution["checkpoint_interval"]),
            {"enabled": bool(self.execution["cache_enabled"]), "force_refresh": False},
            per_record, {}, self.raw, self.config_hash,
        )

    def planned_real_records(self) -> int:
        scenarios = len(self.families) * 3
        return len(self.models) * self.repetitions * scenarios * sum(
            len(self.methods.get(mode, ())) for mode in self.modes
        )
