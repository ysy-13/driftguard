from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from driftguard.contracts.loader import DEFAULT_FIXTURE_PATH, DEFAULT_TASKS_PATH, PROJECT_ROOT


COVERAGE_PATH = PROJECT_ROOT / "benchmark" / "drifts" / "task_drift_coverage_v1.json"
MATCHED_PATH = PROJECT_ROOT / "benchmark" / "matched_failures" / "matched_failures_v1.json"
_PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}|\$\{[^}]+\}")


@dataclass(frozen=True)
class ResolvedTaskInstance:
    public_task_id: str
    source_task_ref: str
    public_scenario_id: str
    instruction: str
    bindings: dict[str, Any]
    actor_id: str
    initial_state_ref: str
    expected_state: tuple[dict[str, Any], ...]
    expected_answer: tuple[dict[str, Any], ...]
    forbidden_side_effects: tuple[dict[str, Any], ...]
    tool_budget: int
    evaluator_ref: str
    oracle_plan: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]

    def evaluator_task(self) -> dict[str, Any]:
        return {
            "task_id": self.public_task_id,
            "instruction": self.instruction,
            "actor_id": self.actor_id,
            "success_assertions": deepcopy(list(self.expected_state)),
            "answer_assertions": deepcopy(list(self.expected_answer)),
            "forbidden_assertions": deepcopy(list(self.forbidden_side_effects)),
            "max_tool_calls": self.tool_budget,
            "bindings": deepcopy(self.bindings),
            "source_task_ref": self.source_task_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskInstanceResolver:
    """Resolve public scenario tasks exclusively from frozen benchmark sources."""

    def __init__(
        self,
        tasks_path: Path | str = DEFAULT_TASKS_PATH,
        coverage_path: Path | str = COVERAGE_PATH,
        fixture_path: Path | str = DEFAULT_FIXTURE_PATH,
    ) -> None:
        self.tasks_path, self.coverage_path, self.fixture_path = map(Path, (tasks_path, coverage_path, fixture_path))
        self.task_document = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        self.coverage_document = json.loads(self.coverage_path.read_text(encoding="utf-8"))
        self.fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self._tasks = {item["task_id"]: item for item in self.task_document["tasks"]}
        self._instances: dict[str, tuple[str, dict[str, Any]]] = {}
        for row in self.coverage_document["coverage"]:
            for key in ("detection_instances", "future_transfer_instances"):
                for instance in row.get(key, ()):
                    self._instances[instance["instance_id"]] = (row["drift_id"], instance)

    def resolve(self, family: dict[str, Any], scenario: dict[str, Any]) -> ResolvedTaskInstance:
        first_failure = next(item for item in scenario["episode_schedule"] if item["phase"] == "first_failure")
        instance_ref = first_failure["task_instance_ref"]
        try:
            drift_id, instance = self._instances[instance_ref]
        except KeyError as exc:
            raise ValueError(f"unresolvable first-failure task instance: {instance_ref}") from exc
        if drift_id != family["source_drift_id"]:
            raise ValueError(f"task instance {instance_ref} belongs to {drift_id}, not the family source")
        source_ref = instance["base_task_id"]
        try:
            task = deepcopy(self._tasks[source_ref])
        except KeyError as exc:
            raise ValueError(f"unresolvable canonical task: {source_ref}") from exc

        replacement_map = self._apply_argument_overrides(task, instance.get("argument_overrides", {}))
        self._replace_assertion_entities(task, replacement_map)
        bindings = self._bindings(task)
        instruction = self._resolved_instruction(instance["instruction"], bindings)
        if _PLACEHOLDER.search(instruction):
            raise ValueError(f"unresolved binding in {instance_ref}: {instruction}")

        scenario_public = scenario["agent_visible"]["evidence_bundle"]["scenario_id"]
        public_task_id = "task-" + hashlib.sha256(instance_ref.encode()).hexdigest()[:12]
        provenance = {
            "family": family["matched_case_id"],
            "task_instance_ref": instance_ref,
            "coverage_ref": f"{self.coverage_path.relative_to(PROJECT_ROOT)}#/{drift_id}/{instance_ref}",
            "canonical_task_ref": f"{self.tasks_path.relative_to(PROJECT_ROOT)}#/tasks/{source_ref}",
            "fixture_ref": f"{self.fixture_path.relative_to(PROJECT_ROOT)}#/{task['initial_state']}",
            "instruction_source": "coverage_detection_instance",
            "evaluator_source": "canonical_task_with_declared_argument_overrides",
            "argument_overrides": deepcopy(instance.get("argument_overrides", {})),
            "fixture_overrides": deepcopy(instance.get("fixture_overrides", {})),
        }
        return ResolvedTaskInstance(
            public_task_id=public_task_id,
            source_task_ref=source_ref,
            public_scenario_id=scenario_public,
            instruction=instruction,
            bindings=bindings,
            actor_id=task["actor_id"],
            initial_state_ref=task["initial_state"],
            expected_state=tuple(deepcopy(task["success_assertions"])),
            expected_answer=tuple(deepcopy(task["answer_assertions"])),
            forbidden_side_effects=tuple(deepcopy(task["forbidden_assertions"])),
            tool_budget=int(task["max_tool_calls"]),
            evaluator_ref=provenance["canonical_task_ref"],
            oracle_plan=tuple(deepcopy(task["oracle_plan"])),
            provenance=provenance,
        )

    @staticmethod
    def _apply_argument_overrides(task: dict[str, Any], overrides: dict[str, Any]) -> dict[tuple[str, Any], Any]:
        replacements: dict[tuple[str, Any], Any] = {}
        omitted = set(overrides.get("omit", ()))
        for step in task["oracle_plan"]:
            arguments = step["arguments"]
            for key in omitted:
                arguments.pop(key, None)
            for key, new_value in overrides.items():
                if key == "omit" or key not in arguments:
                    continue
                old_value = arguments[key]
                arguments[key] = deepcopy(new_value)
                if old_value != new_value:
                    replacements[(key, old_value)] = new_value
        return replacements

    @staticmethod
    def _replace_assertion_entities(task: dict[str, Any], replacements: dict[tuple[str, Any], Any]) -> None:
        for (_, old), new in replacements.items():
            old_token, new_token = str(old), str(new)
            for group in (task["success_assertions"], task["forbidden_assertions"]):
                for assertion in group:
                    if "path" in assertion:
                        assertion["path"] = assertion["path"].replace(f"/{old_token}/", f"/{new_token}/")
                        if assertion["path"].endswith(f"/{old_token}"):
                            assertion["path"] = assertion["path"][: -len(old_token)] + new_token
                    if "allowed_paths" in assertion:
                        assertion["allowed_paths"] = [
                            path.replace(f"/{old_token}/", f"/{new_token}/") for path in assertion["allowed_paths"]
                        ]

    def _bindings(self, task: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for step in task["oracle_plan"]:
            for key, value in step["arguments"].items():
                if not (isinstance(value, str) and value.startswith("$steps.")):
                    values.setdefault(key, deepcopy(value))
                elif key == "issue_id":
                    repo_id = step["arguments"].get("repo_id") or values.get("repo_id")
                    if repo_id in self.fixture.get("next_ids", {}):
                        values["created_issue_id"] = self.fixture["next_ids"][repo_id]["issue_id"]
        repo_id = values.get("repo_id")
        if repo_id in self.fixture.get("repositories", {}):
            values["repository_name"] = self.fixture["repositories"][repo_id]["name"]
        return values

    @staticmethod
    def _resolved_instruction(instruction: str, bindings: dict[str, Any]) -> str:
        repo_id = bindings.get("repo_id")
        repo_name = bindings.get("repository_name")
        suffixes: list[str] = []
        if repo_id and str(repo_id) not in instruction:
            label = f"repository {repo_name!r}" if repo_name else "the target repository"
            suffixes.append(f"{label} has repo_id {repo_id}")
        if "created_issue_id" in bindings:
            suffixes.append("consume the issue_id returned by create_issue for every dependent step")
        if suffixes:
            return instruction.rstrip() + " Resolved identifiers: " + "; ".join(suffixes) + "."
        return instruction


def load_all_resolved_tasks() -> list[ResolvedTaskInstance]:
    resolver = TaskInstanceResolver()
    document = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))
    return [resolver.resolve(family, scenario) for family in document["families"] for scenario in family["scenarios"]]
