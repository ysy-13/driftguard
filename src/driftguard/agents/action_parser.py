from __future__ import annotations

from driftguard.llm.structured_output import parse_structured_output

from .models import AgentAction


class ActionParser:
    def __init__(self, schema: dict):
        self.schema = schema

    def parse(self, raw_text: str) -> AgentAction:
        return AgentAction.from_dict(parse_structured_output(raw_text, self.schema))

