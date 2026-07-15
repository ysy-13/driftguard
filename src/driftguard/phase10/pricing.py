from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm import ModelConfig


class PricingCatalog:
    def __init__(self, document: dict[str, Any]):
        self.document = document
        self.models = {(item["provider"], item["model_id"]): item for item in document["models"]}

    @classmethod
    def load_default(cls) -> "PricingCatalog":
        path = PROJECT_ROOT / "benchmark" / "pricing" / "phase10_pricing_v1.json"
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def estimate_cny(self, config: ModelConfig, input_tokens: int, output_tokens: int) -> float:
        item = self.models[(config.provider, config.model_id)]
        input_cost = input_tokens / 1_000_000 * float(item["input_price_per_million"])
        output_cost = output_tokens / 1_000_000 * float(item["output_price_per_million"])
        amount = input_cost + output_cost
        return amount * float(item.get("usd_cny_estimate", 1.0)) if item["currency"] == "USD" else amount
