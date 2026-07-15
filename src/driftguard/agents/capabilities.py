from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any


BASELINE_ACTIONS = ("TOOL_CALL", "FINAL_ANSWER", "ABSTAIN")
DRIFTGUARD_ACTIONS = (*BASELINE_ACTIONS[:2], "REQUEST_PROBE", "ABSTAIN")


@dataclass(frozen=True)
class PolicyCapabilities:
    method: str
    allowed_actions: tuple[str, ...]

    @classmethod
    def for_method(cls, method: str) -> "PolicyCapabilities":
        actions = DRIFTGUARD_ACTIONS if method.startswith("driftguard_") or method == "oracle_symbolic_upper_bound" else BASELINE_ACTIONS
        return cls(method, actions)

    @property
    def allows_probe(self) -> bool:
        return "REQUEST_PROBE" in self.allowed_actions

    def narrow_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        narrowed = deepcopy(schema)
        branches = narrowed.get("oneOf")
        if not isinstance(branches, list):
            raise ValueError("AgentAction schema must use action-specific oneOf branches")
        narrowed["oneOf"] = [
            branch for branch in branches
            if branch.get("properties", {}).get("action_type", {}).get("const") in self.allowed_actions
        ]
        found = {
            branch["properties"]["action_type"]["const"] for branch in narrowed["oneOf"]
        }
        if found != set(self.allowed_actions):
            raise ValueError(f"schema does not define exactly the allowed actions: {self.allowed_actions}")
        narrowed["title"] = f"AgentAction for {self.method}"
        return narrowed

    def prompt_fragment(self) -> str:
        return (
            "Policy capability contract (authoritative):\n"
            f"Allowed action_type values: {', '.join(self.allowed_actions)}.\n"
            "No action_type omitted from this list or the schema is available."
        )

    def fingerprint(self) -> str:
        value = {"method": self.method, "allowed_actions": self.allowed_actions}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
