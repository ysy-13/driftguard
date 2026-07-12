from __future__ import annotations

from copy import deepcopy
from typing import Any


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def state_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if path == "/clock":
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys):
            child_path = f"{path}/{_escape(str(key))}"
            if child_path != "/clock":
                changes.append({"op": "remove", "path": child_path, "before": deepcopy(before[key])})
        for key in sorted(after_keys - before_keys):
            child_path = f"{path}/{_escape(str(key))}"
            if child_path != "/clock":
                changes.append({"op": "add", "path": child_path, "after": deepcopy(after[key])})
        for key in sorted(before_keys & after_keys):
            changes.extend(state_diff(before[key], after[key], f"{path}/{_escape(str(key))}"))
        return changes
    if before != after:
        return [{"op": "replace", "path": path or "/", "before": deepcopy(before), "after": deepcopy(after)}]
    return []


def path_is_allowed(path: str, allowed_pattern: str) -> bool:
    if allowed_pattern.endswith("/**"):
        base = allowed_pattern[:-3]
        return path == base or path.startswith(base + "/")
    return path == allowed_pattern
