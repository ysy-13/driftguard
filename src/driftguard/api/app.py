from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from driftguard.contracts.registry import ToolContract
from driftguard.sandbox.result import failure
from driftguard.sandbox.service import SandboxService


def _actor_from_request(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return ""
    return token


def _path_arguments(contract: ToolContract, request: Request) -> dict[str, Any]:
    properties = contract.input_schema["properties"]
    arguments: dict[str, Any] = {}
    for name, raw_value in request.path_params.items():
        schema = properties[name]
        arguments[name] = int(raw_value) if schema.get("type") == "integer" else raw_value
    return arguments


def _endpoint(operation_id: str, contract: ToolContract) -> Callable[..., Any]:
    async def call(request: Request) -> JSONResponse:
        arguments = _path_arguments(contract, request)
        body_bytes = await request.body()
        if body_bytes:
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                result = failure(400, "BAD_REQUEST", "Request body must be valid JSON.")
                return JSONResponse(result.payload, status_code=result.status_code)
            if not isinstance(body, dict):
                result = failure(400, "BAD_REQUEST", "Request body must be a JSON object.")
                return JSONResponse(result.payload, status_code=result.status_code)
            arguments.update(body)
        service: SandboxService = request.app.state.sandbox_service
        result = service.call_tool(operation_id, arguments, _actor_from_request(request))
        return JSONResponse(result.payload, status_code=result.status_code)

    call.__name__ = f"route_{operation_id}"
    return call


def create_app(service: SandboxService | None = None) -> FastAPI:
    sandbox_service = service or SandboxService()
    application = FastAPI(title="DriftGuard Local Sandbox", version="1.0.0")
    application.state.sandbox_service = sandbox_service
    for contract in sandbox_service.registry.contracts():
        application.add_api_route(
            contract.path,
            _endpoint(contract.operation_id, contract),
            methods=[contract.method.upper()],
            operation_id=contract.operation_id,
            response_model=None,
        )

    @application.post("/internal/reset", include_in_schema=False)
    async def reset() -> dict[str, bool]:
        sandbox_service.reset()
        return {"ok": True}

    return application


app = create_app()
