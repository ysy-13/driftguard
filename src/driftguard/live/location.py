from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


LOCATION_LAYERS = {"input", "response", "workflow", "state_effect"}


@dataclass(frozen=True)
class NormalizedLocation:
    """One coordinate system shared by attribution, eligibility and patches."""

    tool_id: str
    layer: str
    spec_pointer: str | None
    runtime_path: str | None
    adapter_operation_type: str

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise ValueError("normalized location requires tool_id")
        if self.layer not in LOCATION_LAYERS:
            raise ValueError("unsupported normalized location layer")
        if self.spec_pointer is not None and not self.spec_pointer.startswith("/"):
            raise ValueError("spec_pointer must be an absolute JSON Pointer")
        if not self.spec_pointer and not self.runtime_path:
            raise ValueError("normalized location requires a spec or runtime coordinate")
        if not self.adapter_operation_type:
            raise ValueError("normalized location requires adapter_operation_type")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalizedLocation":
        return cls(
            tool_id=str(value.get("tool_id") or ""),
            layer=str(value.get("layer") or ""),
            spec_pointer=value.get("spec_pointer"),
            runtime_path=value.get("runtime_path"),
            adapter_operation_type=str(value.get("adapter_operation_type") or ""),
        )


def normalize_location(value: Mapping[str, Any]) -> dict[str, Any]:
    return NormalizedLocation.from_mapping(value).to_dict()


def locations_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return normalize_location(left) == normalize_location(right)
