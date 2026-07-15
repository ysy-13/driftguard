from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from driftguard.llm import (
    OpenAICompatibleProvider, ProviderRequest, RequestRateLimiter,
    parse_structured_output,
)
from driftguard.llm.redaction import assert_secret_absent

from .config import Phase10Config
from .costs import CostBudgetManager


JSON_SCHEMA = {
    "type": "object",
    "required": ["status"],
    "properties": {"status": {"const": "ok"}},
    "additionalProperties": False,
}
TOOLS = ({
    "type": "function",
    "function": {
        "name": "phase10_echo",
        "description": "Return the supplied short text.",
        "parameters": {
            "type": "object", "required": ["text"],
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        },
    },
},)


class PreflightRunner:
    def __init__(self, config: Phase10Config, output: Path, costs: CostBudgetManager, provider_builder=None):
        self.config, self.output, self.costs = config, output, costs
        self.provider_builder = provider_builder
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        reports = []
        for model in self.config.models:
            key = os.getenv(model.api_key_env or "")
            if not key:
                reports.append(self._write(model.provider, {
                    "provider": model.provider, "configured_model": model.model_id,
                    "endpoint": model.base_url, "credential": "missing", "passed": False,
                    "checks": [], "sanitized_error": "credential missing",
                }))
                continue
            report = self._run_model(model, key)
            assert_secret_absent(report, key)
            reports.append(self._write(model.provider, report))
        summary = {
            "experiment_stage": "PILOT_PREFLIGHT",
            "reports": reports,
            "passed_models": [item["provider"] for item in reports if item["passed"]],
            "cost": self.costs.snapshot(),
        }
        self._atomic(self.output / "summary.json", summary)
        return summary

    def _run_model(self, model, key: str) -> dict[str, Any]:
        provider = (
            self.provider_builder(model, key)
            if self.provider_builder is not None
            else OpenAICompatibleProvider(
                model, key, rate_limiter=RequestRateLimiter(1), cost_controller=self.costs,
            )
        )
        checks = []
        try:
            text_response = provider.complete(self._request(
                model, "minimal_text", ({"role": "user", "content": "Reply with exactly the word OK."},), {},
            ))
            checks.append(self._check_response("minimal_text", model.model_id, text_response, bool(text_response.raw_text.strip())))

            json_response = provider.complete(self._request(
                model, "strict_json",
                ({"role": "system", "content": "Return only JSON."}, {"role": "user", "content": "Return {\"status\":\"ok\"}."}),
                JSON_SCHEMA,
            ))
            parse_structured_output(json_response.raw_text, JSON_SCHEMA)
            check = self._check_response("strict_json", model.model_id, json_response, True)
            check["structured_output_mode_used"] = model.structured_output_mode
            checks.append(check)

            tool_response = provider.complete(ProviderRequest(
                ({"role": "user", "content": "Call phase10_echo with text set to ok."},), {},
                hashlib.sha256(b"phase10-tool-preflight").hexdigest(),
                "public-preflight0001", 1, "preflight_tool", 0,
                self.config.seeds[0], "preflight", self.config.config_hash, TOOLS, "required",
            ))
            checks.append(self._check_response(
                "tool_calling", model.model_id, tool_response, bool(tool_response.tool_calls),
            ))
        except Exception as exc:
            return {
                "provider": model.provider, "configured_model": model.model_id,
                "endpoint": model.base_url, "credential": "configured", "passed": False,
                "checks": checks, "sanitized_error": f"{type(exc).__name__}: {exc}",
                "thinking_mode": model.thinking_mode,
            }
        return {
            "provider": model.provider, "configured_model": model.model_id,
            "endpoint": model.base_url, "credential": "configured",
            "passed": all(item["passed"] for item in checks), "checks": checks,
            "sanitized_error": None, "thinking_mode": model.thinking_mode,
        }

    def _request(self, model, method, messages, schema):
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        return ProviderRequest(
            tuple(messages), schema, prompt_hash, "public-preflight0001", 1, method, 0,
            self.config.seeds[0], "preflight", self.config.config_hash,
        )

    @staticmethod
    def _check_response(name, configured_model, response, content_ok):
        usage_ok = response.total_tokens > 0 and response.input_tokens > 0
        model_ok = response.model == configured_model
        return {
            "name": name, "passed": bool(content_ok and usage_ok and model_ok),
            "actual_model": response.model, "model_match": model_ok,
            "usage_returned": usage_ok, "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens, "provider_attempts": response.provider_attempts,
        }

    def _write(self, provider, report):
        self._atomic(self.output / f"{provider}.json", report)
        return report

    @staticmethod
    def _atomic(path: Path, value: dict[str, Any]):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
