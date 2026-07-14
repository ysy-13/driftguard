from __future__ import annotations

from .models import PatchLifecycle, ToolSpecPatch


class PatchRegistry:
    def __init__(self) -> None:
        self._patches: dict[tuple[str, str, str, str], ToolSpecPatch] = {}

    @staticmethod
    def key(patch: ToolSpecPatch) -> tuple[str, str, str, str]:
        return (patch.target_tool_id, patch.source_spec_fingerprint, patch.location_path, patch.patch_version)

    def register(self, patch: ToolSpecPatch) -> None:
        if patch.lifecycle_status != PatchLifecycle.ACCEPTED:
            raise ValueError("only accepted patches may be registered")
        self._patches[self.key(patch)] = patch

    def get(self, target_tool_id: str, source_fingerprint: str, location_path: str, version: str = "1.0") -> ToolSpecPatch | None:
        return self._patches.get((target_tool_id, source_fingerprint, location_path, version))

    def __len__(self) -> int:
        return len(self._patches)

