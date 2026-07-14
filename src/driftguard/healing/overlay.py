from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.evidence.collector import stable_hash

from .models import ToolSpecPatch
from .patch_document import PatchApplicationError, apply_operation


class SpecOverlay:
    def __init__(self, displayed_spec: dict[str, Any]):
        self._source = deepcopy(displayed_spec)
        self._document = deepcopy(displayed_spec)
        self.source_fingerprint = stable_hash(displayed_spec)

    @property
    def document(self) -> dict[str, Any]:
        return deepcopy(self._document)

    def apply(self, patch: ToolSpecPatch) -> dict[str, Any]:
        if patch.source_spec_fingerprint != self.source_fingerprint:
            raise PatchApplicationError("source specification fingerprint mismatch")
        candidate = deepcopy(self._document)
        for operation in patch.openapi_operations:
            apply_operation(candidate, operation.to_dict())
        self._document = candidate
        return self.document

    def rollback(self) -> None:
        self._document = deepcopy(self._source)

    def serialize(self) -> dict[str, Any]:
        return self.document

