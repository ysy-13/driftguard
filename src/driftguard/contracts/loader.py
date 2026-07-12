from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPENAPI_PATH = PROJECT_ROOT / "benchmark" / "openapi" / "driftguard_openapi_v1.yaml"
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "benchmark" / "fixtures" / "initial_state_v1.json"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "benchmark" / "tasks" / "tasks_v1.json"


def load_openapi(path: Path | str = DEFAULT_OPENAPI_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("OpenAPI root must be a mapping")
    return document


def resolve_ref(document: dict[str, Any], value: Any) -> Any:
    seen: set[str] = set()
    while isinstance(value, dict) and set(value) == {"$ref"}:
        ref = value["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError(f"unsupported reference: {ref!r}")
        if ref in seen:
            raise ValueError(f"circular reference: {ref}")
        seen.add(ref)
        current: Any = document
        for raw_token in ref[2:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, dict) or token not in current:
                raise ValueError(f"unresolvable reference: {ref}")
            current = current[token]
        value = current
    return value


def deep_resolve(document: dict[str, Any], value: Any) -> Any:
    value = resolve_ref(document, value)
    if isinstance(value, dict):
        return {key: deep_resolve(document, child) for key, child in value.items()}
    if isinstance(value, list):
        return [deep_resolve(document, child) for child in value]
    return value
