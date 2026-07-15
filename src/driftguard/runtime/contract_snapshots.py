from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from driftguard.contracts.loader import DEFAULT_OPENAPI_PATH, load_openapi
from .execution_profile import ExecutionMode, ExecutionProfile


@dataclass(frozen=True)
class ContractSnapshot:
    source: str
    document: dict[str, Any]
    overlays: tuple[dict[str, Any], ...] = ()

    def copy_document(self) -> dict[str, Any]:
        return deepcopy(self.document)

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _resolve_parent(document: dict[str, Any], pointer: str) -> tuple[Any, str]:
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer.strip("/").split("/")]
    current: Any = document
    for token in tokens[:-1]:
        current = current[token]
    return current, tokens[-1]


def _mutate_contract(canonical: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    document = deepcopy(canonical)
    mutation = case["runtime_mutation"]
    operation = mutation["operation"]
    parent, key = _resolve_parent(document, mutation["target"])
    before, after = mutation["before"], mutation["after"]
    if operation == "add_required":
        if isinstance(after, list):
            parent[key] = deepcopy(after)
            for field in set(after) - set(before):
                parent.get("properties", {}).get(field, {}).pop("default", None)
        else:
            parent[key].pop("default", None)
            schema_pointer = mutation["target"].rsplit("/properties/", 1)[0]
            schema_parent, schema_key = _resolve_parent(document, schema_pointer)
            required = schema_parent[schema_key].setdefault("required", [])
            if key not in required:
                required.append(key)
    elif operation == "rename_input_field":
        schema_pointer = mutation["target"].rsplit("/properties/", 1)[0]
        schema_parent, schema_key = _resolve_parent(document, schema_pointer)
        schema = schema_parent[schema_key]
        schema["properties"][after] = schema["properties"].pop(before)
        schema["required"] = [after if item == before else item for item in schema.get("required", [])]
    elif operation == "replace_enum_value":
        parent[key] = deepcopy(after)
    elif operation == "rename_output_field":
        parent[after] = parent.pop(before)
        required_pointer = mutation["target"].rsplit("/properties/", 1)[0]
        req_parent, req_key = _resolve_parent(document, required_pointer)
        schema = req_parent[req_key]
        schema["required"] = [after if item == before else item for item in schema.get("required", [])]
    elif operation == "map_output_path":
        for old, new in zip(before, after):
            parent[key][new] = parent[key].pop(old)
    elif operation == "replace_output_path":
        properties = parent[key]
        nested: dict[str, Any] = {}
        for old, path in zip(before, after):
            nested[path.split(".")[-1]] = properties.pop(old)
        properties["permissions"] = {"type": "object", "properties": nested, "required": list(nested)}
    else:
        parent[key] = deepcopy(after)
    return document


class CanonicalContractSnapshotManager:
    def __init__(self) -> None:
        self._canonical = load_openapi()
        self._file_hash = hashlib.sha256(DEFAULT_OPENAPI_PATH.read_bytes()).hexdigest()

    @property
    def canonical_sha256(self) -> str:
        return self._file_hash

    def canonical_copy(self) -> dict[str, Any]:
        return deepcopy(self._canonical)

    def mutated_copy(self, case: dict[str, Any]) -> dict[str, Any]:
        return _mutate_contract(self._canonical, case)


@dataclass(frozen=True)
class ContractSnapshots:
    displayed: ContractSnapshot
    runtime: ContractSnapshot
    alignment_mode: str

    @classmethod
    def for_profile(cls, profile: ExecutionProfile, episode_index: int = 3) -> "ContractSnapshots":
        manager = CanonicalContractSnapshotManager()
        canonical = manager.canonical_copy()
        mutation = deepcopy(profile.drift_case["runtime_mutation"])
        canonical_displayed = ContractSnapshot("canonical_v1", deepcopy(canonical))
        canonical_runtime = ContractSnapshot("canonical_v1", deepcopy(canonical))
        if profile.mode == ExecutionMode.AGENT_ERROR:
            mutated = manager.mutated_copy(profile.drift_case)
            changed = ContractSnapshot("canonical_v1_plus_runtime_rule", mutated, (mutation,))
            return cls(changed, ContractSnapshot(changed.source, deepcopy(mutated), (deepcopy(mutation),)), "aligned_new")
        if profile.mode == ExecutionMode.PERSISTENT_DRIFT and profile.runtime_contract_active(episode_index):
            mutated = manager.mutated_copy(profile.drift_case)
            return cls(canonical_displayed, ContractSnapshot("runtime_drift", mutated, (mutation,)), "displayed_old_runtime_new")
        return cls(canonical_displayed, canonical_runtime, "aligned_canonical")

    @property
    def contracts_equal(self) -> bool:
        return self.displayed.document == self.runtime.document and self.displayed.overlays == self.runtime.overlays
