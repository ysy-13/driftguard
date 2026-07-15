from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ALLOWED_KEY_NAMES = ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")


def load_project_dotenv(project_root: Path) -> None:
    """Load only the two Phase 10 credentials without exposing their values."""
    path = project_root / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or ("=" not in line and ":" not in line):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        delimiter = "=" if "=" in line else ":"
        name, value = line.split(delimiter, 1)
        name = name.strip().strip('"\'')
        if name not in ALLOWED_KEY_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def credential_status(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    return {name: "configured" if bool(env.get(name)) else "missing" for name in ALLOWED_KEY_NAMES}


def safe_status_lines(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    return tuple(f"{name}: {status}" for name, status in credential_status(environ).items())
