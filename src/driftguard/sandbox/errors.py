from __future__ import annotations


class SandboxError(Exception):
    status_code = 400
    code = "BAD_REQUEST"

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class NotFoundError(SandboxError):
    status_code = 404
    code = "NOT_FOUND"


class PreconditionError(SandboxError):
    status_code = 409
    code = "PRECONDITION_FAILED"
