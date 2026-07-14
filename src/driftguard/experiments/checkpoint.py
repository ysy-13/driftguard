from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any


class CheckpointStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.records_dir = self.directory / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def has(self, record_id: str) -> bool:
        return (self.records_dir / f"{record_id}.json").exists()

    def read(self, record_id: str) -> dict[str, Any]:
        return json.loads((self.records_dir / f"{record_id}.json").read_text(encoding="utf-8"))

    def write(self, record_id: str, record: dict[str, Any]) -> None:
        final = self.records_dir / f"{record_id}.json"
        temporary = self.records_dir / f".{record_id}.tmp"
        with self._lock:
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(final)

    def all_records(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.records_dir.glob("*.json"))]

    def write_summary(self, document: dict[str, Any]) -> None:
        final, temporary = self.directory / "summary.json", self.directory / ".summary.tmp"
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(final)

