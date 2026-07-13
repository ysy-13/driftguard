from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from .pending_effects import PendingEffects


@dataclass
class VerificationBinding:
    kind: str
    repo_id: str
    resource_id: str
    episode_index: int
    used: bool = False


@dataclass
class RuntimeSessionState:
    verification_tokens: dict[str, VerificationBinding] = field(default_factory=dict)
    pending_effects: PendingEffects = field(default_factory=PendingEffects)
    consumed_transient: set[str] = field(default_factory=set)
    call_count: int = 0
    recent_read: dict[str, Any] | None = None
    response_interpretation: dict[str, Any] | None = None

    def issue_token(
        self,
        session_key: str,
        kind: str,
        repo_id: str,
        resource_id: str,
        episode_index: int,
    ) -> str:
        material = f"{session_key}|{kind}|{repo_id}|{resource_id}|{episode_index}".encode()
        token = f"verify_{hashlib.sha256(material).hexdigest()[:24]}"
        self.verification_tokens[token] = VerificationBinding(
            kind=kind,
            repo_id=repo_id,
            resource_id=str(resource_id),
            episode_index=episode_index,
        )
        self.recent_read = {"kind": kind, "repo_id": repo_id, "resource_id": str(resource_id)}
        return token

    def consume_token(
        self,
        token: str | None,
        kind: str,
        repo_id: str,
        resource_id: str,
        episode_index: int,
    ) -> bool:
        binding = self.verification_tokens.get(token or "")
        valid = bool(
            binding
            and not binding.used
            and binding.kind == kind
            and binding.repo_id == repo_id
            and binding.resource_id == str(resource_id)
            and binding.episode_index == episode_index
        )
        if valid and binding is not None:
            binding.used = True
        return valid

    def reset(self) -> None:
        self.verification_tokens.clear()
        self.pending_effects.clear()
        self.consumed_transient.clear()
        self.call_count = 0
        self.recent_read = None
        self.response_interpretation = None
