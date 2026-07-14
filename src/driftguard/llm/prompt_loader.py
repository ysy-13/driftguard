from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from driftguard.evidence.leakage_guard import assert_agent_visible


class PromptLoader:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self, name: str) -> str:
        text = (self.root / name).read_text(encoding="utf-8")
        assert_agent_visible(text)
        if re.search(r"\bM\d{2}\b|\b(?:ICD|RSD|WPD|SED)-\d{2}\b", text, re.IGNORECASE):
            raise ValueError("prompt contains a benchmark family or drift identifier")
        return text

    def render(self, name: str, values: dict[str, Any]) -> str:
        assert_agent_visible(values)
        return self.load(name).format(**values)

    def hash(self, name: str) -> str:
        return hashlib.sha256(self.load(name).encode()).hexdigest()
