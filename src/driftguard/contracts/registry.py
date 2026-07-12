from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .loader import DEFAULT_OPENAPI_PATH, deep_resolve, load_openapi, resolve_ref


HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


@dataclass(frozen=True)
class ToolContract:
    operation_id: str
    method: str
    path: str
    input_schema: dict[str, Any]
    body_schema: dict[str, Any] | None
    path_parameters: tuple[str, ...]
    body_properties: tuple[str, ...]
    request_body_required: bool
    required_permission: str
    effect_type: str
    idempotent: bool | str
    preconditions: tuple[str, ...]
    state_effects: tuple[str, ...]
    success_status_code: int
    success_schema: dict[str, Any]


class ContractRegistry:
    def __init__(self, document: dict[str, Any]):
        self.document = document
        self._contracts = self._build_contracts()

    @classmethod
    def from_openapi(cls, path: Path | str = DEFAULT_OPENAPI_PATH) -> "ContractRegistry":
        return cls(load_openapi(path))

    def _operations(self) -> Iterator[tuple[str, str, dict[str, Any]]]:
        for path, path_item in self.document.get("paths", {}).items():
            if not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                    yield path, method.lower(), operation

    def _build_contracts(self) -> dict[str, ToolContract]:
        contracts: dict[str, ToolContract] = {}
        for path, method, operation in self._operations():
            path_properties: dict[str, Any] = {}
            path_required: list[str] = []
            for raw_parameter in operation.get("parameters", []):
                parameter = resolve_ref(self.document, raw_parameter)
                name = parameter["name"]
                path_properties[name] = deep_resolve(self.document, parameter["schema"])
                if parameter.get("required"):
                    path_required.append(name)

            body_schema = None
            body_properties: dict[str, Any] = {}
            body_required: list[str] = []
            request_body = operation.get("requestBody")
            if request_body:
                body_schema = deep_resolve(
                    self.document,
                    request_body["content"]["application/json"]["schema"],
                )
                body_properties = body_schema.get("properties", {})
                body_required = list(body_schema.get("required", []))
            flat_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {**path_properties, **body_properties},
                "required": path_required + body_required,
            }
            success_status = min(
                int(code)
                for code in operation["responses"]
                if str(code).isdigit() and str(code).startswith("2")
            )
            response = operation["responses"][str(success_status)]
            success_schema = deep_resolve(
                self.document,
                response["content"]["application/json"]["schema"],
            )
            operation_id = operation["operationId"]
            if operation_id in contracts:
                raise ValueError(f"duplicate operationId: {operation_id}")
            contracts[operation_id] = ToolContract(
                operation_id=operation_id,
                method=method,
                path=path,
                input_schema=flat_schema,
                body_schema=body_schema,
                path_parameters=tuple(path_properties),
                body_properties=tuple(body_properties),
                request_body_required=bool(request_body and request_body.get("required")),
                required_permission=operation["x-required-permission"],
                effect_type=operation["x-effect-type"],
                idempotent=operation["x-idempotent"],
                preconditions=tuple(operation["x-preconditions"]),
                state_effects=tuple(operation["x-state-effects"]),
                success_status_code=success_status,
                success_schema=success_schema,
            )
        return contracts

    def get(self, operation_id: str) -> ToolContract | None:
        return self._contracts.get(operation_id)

    def require(self, operation_id: str) -> ToolContract:
        contract = self.get(operation_id)
        if contract is None:
            raise KeyError(operation_id)
        return contract

    def operation_ids(self) -> tuple[str, ...]:
        return tuple(self._contracts)

    def contracts(self) -> tuple[ToolContract, ...]:
        return tuple(self._contracts.values())
