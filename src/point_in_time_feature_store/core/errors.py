"""Error responses that never leak internals.

Every error body has the same shape: ``{"error": {"code", "message", "request_id"}}``. Messages
are fixed strings chosen by the service. Validation errors report location, type and pydantic's
message but never the rejected input, and unexpected exceptions become a generic 500 (the
traceback goes to the redacting logger, not the client).
"""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

__all__ = [
    "AppError",
    "BodyTooLargeError",
    "error_payload",
    "install_error_handlers",
    "request_id_of",
]

_STATUS_CODES: Mapping[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    429: "rate_limited",
}


class AppError(Exception):
    """An expected, client-visible error with a stable machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


class BodyTooLargeError(StarletteHTTPException):
    """Raised while streaming a request body past the configured limit.

    It subclasses Starlette's HTTPException so FastAPI's body parser re-raises it as-is instead of
    converting it into a generic 400.
    """

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Request body too large")


def request_id_of(request: Request) -> str | None:
    state = request.scope.get("state")
    if isinstance(state, Mapping):
        value = state.get("request_id")
        return value if isinstance(value, str) else None
    return None


def error_payload(code: str, message: str, request_id: str | None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):  # registered for AppError only
        raise exc
    return JSONResponse(
        error_payload(exc.code, exc.message, request_id_of(request)),
        status_code=exc.status_code,
        headers=exc.headers,
    )


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # registered for HTTPException only
        raise exc
    status = exc.status_code
    code = _STATUS_CODES.get(status, f"http_{status}")
    try:
        message = HTTPStatus(status).phrase
    except ValueError:
        message = "Request failed"
    headers = dict(exc.headers) if exc.headers else None
    return JSONResponse(
        error_payload(code, message, request_id_of(request)), status_code=status, headers=headers
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # registered for validation only
        raise exc
    details = [
        {
            "loc": [str(part)[:64] for part in err.get("loc", ())],
            "type": str(err.get("type", "invalid")),
            "msg": str(err.get("msg", "invalid value"))[:200],
        }
        for err in exc.errors()[:20]
    ]
    body = error_payload("validation_error", "Request validation failed", request_id_of(request))
    body["error"]["details"] = details
    return JSONResponse(body, status_code=422)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
