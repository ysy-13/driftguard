from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm.redaction import assert_secret_absent
from driftguard.runners.injection_conformance_runner import PROTECTED_PATHS

from .config import ExperimentConfig


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(config: ExperimentConfig, prompt_hashes: dict[str, str], api_key: str | None = None) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip())
    dependencies = {}
    for package in ("PyYAML", "jsonschema", "httpx", "pytest", "fastapi"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    manifest = {
        "experiment_id": config.experiment_id,
        "timestamp": "2026-01-01T00:00:00Z" if config.model.provider == "mock" else __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "git_commit": commit, "dirty_working_tree": dirty,
        "dirty_warning": "Working tree was dirty at experiment start." if dirty else None,
        "config_hash": config.config_hash, "prompt_hashes": prompt_hashes,
        "benchmark_hashes": {name: file_hash(PROJECT_ROOT / name) for name in PROTECTED_PATHS},
        "model_configuration": config.model.public_dict(),
        "supported_model_capabilities": config.model.capabilities.to_dict(),
        "methods": list(config.methods), "scenario_selection": {"families": list(config.families), "modes": list(config.modes)},
        "repetitions": config.repetitions, "budgets": config.budget.to_dict(), "cache_policy": config.cache,
        "execution_controls": {
            "parallelism": config.parallelism,
            "rate_limit_per_second": config.rate_limit_per_second,
            "checkpoint_interval": config.checkpoint_interval,
        },
        "python_version": platform.python_version(), "dependency_versions": dependencies,
        "random_seed": config.seeds[0],
        "real_llm_experiment_status": "NOT RUN" if config.model.provider == "mock" else "RUN",
        "mock_provider": config.model.provider == "mock",
    }
    assert_secret_absent(manifest, api_key)
    return manifest


def write_manifest_once(directory: Path, manifest: dict[str, Any]) -> None:
    path = directory / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["config_hash"] != manifest["config_hash"]:
            raise ValueError("resume manifest config hash mismatch")
        return
    temporary = directory / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
