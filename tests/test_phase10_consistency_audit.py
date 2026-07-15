from __future__ import annotations

import json

from driftguard.agents.action_parser import ActionParser
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.consistency_audit import (
    agent_fault_scope_matrix,
    required_default_negative_audit,
    source_snapshot,
)


def test_required_default_negative_cases_all_pass():
    rows = required_default_negative_audit()
    assert len(rows) == 5
    assert all(row["passed"] for row in rows)


def test_agent_fault_scope_is_variant_provider_and_method_invariant():
    rows = agent_fault_scope_matrix()
    assert len(rows) == 30
    assert all(row["passed"] for row in rows)
    assert sum(row["active_at_protocol_episode"] for row in rows) == 10
    assert not any(row["active_next_episode"] for row in rows)


def test_source_snapshot_is_stable_and_complete():
    first = source_snapshot()
    assert first == source_snapshot()
    assert len(first["files"]) >= 18


def test_action_parser_rejects_semantic_aliases():
    # Existing strict schema must not silently repair model-produced aliases.
    schema = json.loads(
        (PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v3.json").read_text(encoding="utf-8")
    )
    action_parser = ActionParser(schema)
    for raw in (
        '{"action_type":"TOOL_CALL","tool":"create_issue","parameters":{},"concise_decision_summary":"x"}',
        '{"action_type":"TOOL_CALL","tool_id":"create_issue","parameters":{},"concise_decision_summary":"x"}',
    ):
        try:
            action_parser.parse(raw)
        except Exception:
            pass
        else:
            raise AssertionError("semantic action aliases must be rejected")
