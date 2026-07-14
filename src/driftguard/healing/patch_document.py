from __future__ import annotations

from copy import deepcopy
from typing import Any


class PatchApplicationError(ValueError):
    pass


def decode_pointer(path: str) -> list[str]:
    if not path.startswith("/"):
        raise PatchApplicationError("invalid JSON Pointer")
    if path == "/":
        return [""]
    return [token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")]


def _parent(document: Any, path: str) -> tuple[Any, str]:
    tokens = decode_pointer(path)
    if not tokens:
        raise PatchApplicationError("root replacement is forbidden")
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise PatchApplicationError(f"invalid array pointer: {path}") from exc
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise PatchApplicationError(f"missing JSON Pointer parent: {path}")
    return current, tokens[-1]


def apply_operation(document: dict[str, Any], operation: dict[str, Any]) -> None:
    op, path = operation["op"], operation["path"]
    parent, token = _parent(document, path)
    if op == "add":
        if isinstance(parent, list):
            if token == "-":
                parent.append(deepcopy(operation.get("value")))
            else:
                parent.insert(int(token), deepcopy(operation.get("value")))
        elif isinstance(parent, dict):
            parent[token] = deepcopy(operation.get("value"))
        else:
            raise PatchApplicationError(f"cannot add at {path}")
    elif op in {"replace", "test", "remove"}:
        if isinstance(parent, list):
            index = int(token)
            if index >= len(parent):
                raise PatchApplicationError(f"missing JSON Pointer: {path}")
            current = parent[index]
            if op == "replace":
                parent[index] = deepcopy(operation.get("value"))
            elif op == "remove":
                parent.pop(index)
            elif current != operation.get("value"):
                raise PatchApplicationError(f"test operation failed: {path}")
        elif isinstance(parent, dict) and token in parent:
            current = parent[token]
            if op == "replace":
                parent[token] = deepcopy(operation.get("value"))
            elif op == "remove":
                del parent[token]
            elif current != operation.get("value"):
                raise PatchApplicationError(f"test operation failed: {path}")
        else:
            raise PatchApplicationError(f"missing JSON Pointer: {path}")
    else:
        raise PatchApplicationError(f"operation not implemented in v1: {op}")

