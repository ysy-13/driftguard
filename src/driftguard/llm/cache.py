from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from .configuration import ModelConfig
from .models import ProviderRequest, ProviderResponse
from .redaction import assert_secret_absent
from .leakage import assert_provider_request_visible


class LLMCache:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.raw_directory = self.directory / "raw"
        self.parsed_directory = self.directory / "parsed"
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.parsed_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def key(config: ModelConfig, request: ProviderRequest, response_schema_hash: str) -> str:
        assert_provider_request_visible(request)
        material = {
            "provider": config.provider, "model_id": config.model_id,
            "model_configuration": config.public_dict(), "prompt_hash": request.prompt_hash,
            "response_schema_hash": response_schema_hash,
            "public_scenario_id": request.public_scenario_id, "episode": request.episode,
            "method": request.method, "repetition": request.repetition,
            "seed": request.seed, "mode": request.mode, "config_hash": request.config_hash,
            "message_hash": hashlib.sha256(json.dumps([dict(item) for item in request.messages], sort_keys=True).encode()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def get(self, key: str) -> ProviderResponse | None:
        path = self.raw_directory / f"{key}.json"
        parsed_path = self.parsed_directory / f"{key}.json"
        with self._lock:
            if not path.exists():
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            value["parsed_output"] = json.loads(parsed_path.read_text(encoding="utf-8")) if parsed_path.exists() else None
        return ProviderResponse(**value, cached=True)

    def put(self, key: str, response: ProviderResponse, api_key: str | None = None) -> None:
        value = response.public_dict()
        value.pop("cached", None)
        parsed_output = value.pop("parsed_output", None)
        assert_secret_absent(value, api_key)
        path, temporary = self.raw_directory / f"{key}.json", self.raw_directory / f".{key}.tmp"
        parsed_path, parsed_temporary = self.parsed_directory / f"{key}.json", self.parsed_directory / f".{key}.tmp"
        with self._lock:
            temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            parsed_temporary.write_text(json.dumps(parsed_output, sort_keys=True) + "\n", encoding="utf-8")
            parsed_temporary.replace(parsed_path)
