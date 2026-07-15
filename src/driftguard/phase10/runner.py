from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from driftguard.contracts.loader import PROJECT_ROOT

from .config import Phase10Config
from .costs import CostBudgetManager
from .credentials import credential_status, load_project_dotenv
from .pilot import Phase10PilotHarness
from .preflight import PreflightRunner
from .pricing import PricingCatalog


class Phase10Execution:
    def __init__(
        self,
        config_path: Path,
        output: Path | None,
        allow_real_api: bool,
        confirm_full_run: bool = False,
    ):
        load_project_dotenv(PROJECT_ROOT)
        self.config = Phase10Config.load(config_path)
        self.root = Path(output or PROJECT_ROOT / "results/experiments/phase10")
        self.root.mkdir(parents=True, exist_ok=True)
        if not allow_real_api:
            raise PermissionError("Phase 10 real requests require explicit --allow-real-api")
        if self.config.stage != "PILOT":
            if not confirm_full_run:
                raise PermissionError("main/ablation execution requires separate --confirm-full-run authorization")
            raise PermissionError("Full real-model experiment status: NOT RUN; this Phase 10A command cannot execute it")
        self.status = credential_status()
        ledger = self._read_ledger()
        self.costs = CostBudgetManager(
            PricingCatalog.load_default(),
            self.config.budget.pilot_soft_limit_cny,
            self.config.budget.pilot_hard_limit_cny,
            self.config.budget.max_input_tokens_per_record,
            self.config.budget.max_output_tokens_per_record,
            ledger.get("spent_cny", 0.0), ledger.get("api_attempts", 0),
            ledger.get("provider_reported_input_tokens", 0),
            ledger.get("provider_reported_output_tokens", 0),
        )

    def run_preflight(self) -> dict[str, Any]:
        path = self.root / "preflight/summary.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if len(existing.get("passed_models", [])) == len(self.config.models):
                return existing
        result = PreflightRunner(self.config, self.root / "preflight", self.costs).run()
        self._write_ledger()
        return result

    def run_pilot(self, resume: bool = False) -> dict[str, Any]:
        attempt = str(self.config.raw.get("experiment", {}).get("pilot_attempt", "v1"))
        directory = self.root / ("pilot" if attempt == "v1" else f"pilot_{attempt}")
        result = Phase10PilotHarness(self.config, directory, self.costs).run(resume)
        self._write_ledger()
        return result

    def run_component_canary(self, provider: str) -> dict[str, Any]:
        if self.config.raw.get("experiment", {}).get("run_scope") == "end_to_end_only":
            raise PermissionError("this attempt is end-to-end only; component records must not be rerun")
        attempt = str(self.config.raw.get("experiment", {}).get("pilot_attempt", "v1"))
        directory = self.root / ("pilot" if attempt == "v1" else f"pilot_{attempt}")
        harness = Phase10PilotHarness(self.config, directory, self.costs)
        preflight = harness._preflight_summary()
        if provider not in preflight.get("passed_models", []):
            raise PermissionError(f"{provider} did not pass preflight")
        harness._write_manifest()
        harness._ensure_prompt_freeze_before()
        model = next(item for item in self.config.models if item.provider == provider)
        _, report = harness._run_stage(model, "component", resume=True, canary=True)
        self._write_ledger()
        return report

    def run_end_to_end_canary(self, provider: str) -> dict[str, Any]:
        if self.config.raw.get("experiment", {}).get("run_scope") != "end_to_end_only":
            raise PermissionError("end-to-end canary requires an end_to_end_only Pilot attempt")
        attempt = str(self.config.raw.get("experiment", {}).get("pilot_attempt", "v1"))
        directory = self.root / ("pilot" if attempt == "v1" else f"pilot_{attempt}")
        harness = Phase10PilotHarness(self.config, directory, self.costs)
        preflight = harness._preflight_summary()
        if provider not in preflight.get("passed_models", []):
            raise PermissionError(f"{provider} did not pass preflight")
        harness._write_manifest()
        harness._ensure_prompt_freeze_before()
        model = next(item for item in self.config.models if item.provider == provider)
        _, report = harness._run_stage(
            model, "end_to_end", resume=True, canary=True,
        )
        self._write_ledger()
        return report

    def safe_credential_lines(self) -> tuple[str, ...]:
        return tuple(f"{name}: {value}" for name, value in self.status.items())

    def _read_ledger(self) -> dict[str, Any]:
        path = self.root / "cost/ledger.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _write_ledger(self) -> None:
        path = self.root / "cost/ledger.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"config_hash": self.config.config_hash, **self.costs.snapshot()}
        temporary = path.with_name(".ledger.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
