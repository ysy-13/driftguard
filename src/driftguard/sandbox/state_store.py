from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from driftguard.contracts.loader import DEFAULT_FIXTURE_PATH


class StateStore:
    def __init__(self, initial_state: dict[str, Any]):
        self._initial_state = deepcopy(initial_state)
        self._state = deepcopy(initial_state)

    @classmethod
    def from_fixture(cls, path: Path | str = DEFAULT_FIXTURE_PATH) -> "StateStore":
        with Path(path).open(encoding="utf-8") as stream:
            state = json.load(stream)
        if not isinstance(state, dict):
            raise ValueError("fixture root must be an object")
        return cls(state)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "StateStore":
        return cls(state)

    @property
    def initial_clock(self) -> str:
        return str(self._initial_state["clock"])

    def reset(self) -> None:
        self._state = deepcopy(self._initial_state)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def commit(self, new_state: dict[str, Any]) -> None:
        self._state = deepcopy(new_state)

    def get_repository(self, repo_id: str) -> dict[str, Any] | None:
        repository = self._state["repositories"].get(repo_id)
        return deepcopy(repository) if repository is not None else None

    def allocate_issue_id(self, repo_id: str) -> int:
        value = self._state["next_ids"][repo_id]["issue_id"]
        self._state["next_ids"][repo_id]["issue_id"] += 1
        return value

    def allocate_run_id(self, repo_id: str) -> int:
        value = self._state["next_ids"][repo_id]["run_id"]
        self._state["next_ids"][repo_id]["run_id"] += 1
        return value
