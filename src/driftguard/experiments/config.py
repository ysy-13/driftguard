from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml

from driftguard.llm import ModelConfig
from .budgets import ExperimentBudget


ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig
    families: tuple[str, ...]
    methods: tuple[str, ...]
    modes: tuple[str, ...]
    repetitions: int
    seeds: tuple[int, ...]
    parallelism: int
    rate_limit_per_second: float | None
    checkpoint_interval: int
    cache: dict[str, Any]
    budget: ExperimentBudget
    ablations: dict[str, bool]
    raw: dict[str, Any]
    config_hash: str

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        raw = _resolve_environment(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        if not isinstance(raw, dict):
            raise ValueError("experiment config must be an object")
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        budget = ExperimentBudget(**raw.get("budgets", {}))
        return cls(
            str(raw["experiment_id"]), ModelConfig.from_mapping(raw.get("provider", {})),
            tuple(raw.get("families", ())), tuple(raw.get("methods", ())), tuple(raw.get("modes", ())),
            int(raw.get("repetitions", 1)), tuple(int(seed) for seed in raw.get("seeds", [7])),
            int(raw.get("parallelism", 1)),
            None if raw.get("rate_limit_per_second") is None else float(raw["rate_limit_per_second"]),
            int(raw.get("checkpoint_interval", 1)),
            dict(raw.get("cache", {})), budget,
            {key: bool(value) for key, value in raw.get("ablations", {}).items()},
            raw, hashlib.sha256(encoded.encode()).hexdigest(),
        )


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(child) for child in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            return os.environ.get(match.group(1), "")
    return value
