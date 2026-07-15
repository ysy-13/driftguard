from __future__ import annotations

from copy import deepcopy

from driftguard.llm.structured_output import parse_structured_output

from .models import AgentAction


class ActionParser:
    def __init__(self, schema: dict):
        self.schema = schema
        self.last_raw_action: dict | None = None

    def parse(self, raw_text: str) -> AgentAction:
        parsed = parse_structured_output(raw_text, self.schema)
        self.last_raw_action = deepcopy(parsed)
        return AgentAction.from_dict(parsed)
