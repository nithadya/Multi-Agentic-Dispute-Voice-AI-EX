"""
HTTP middleware for the DisputeAI E-Commerce Specialist API.

Adds three pieces of cross-cutting behaviour to every request:

1. ``X-Request-Id`` — generated if the client didn't supply one; echoed
   back on the response for log correlation.
2. ``X-Latency-Ms`` — server-side wall-clock time per request.
3. Unhandled-exception handler — turns any uncaught error into a clean
   JSON 500 with the request id, instead of Starlette's default HTML.
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Stamps every request/response with an id and measures wall-clock latency."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = req_id

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        response.headers["x-request-id"] = req_id
        response.headers["x-latency-ms"] = str(latency_ms)
        return response


def install_middleware(app: FastAPI) -> None:
    """Attach middleware + the catch-all exception handler."""
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error on {} {} [req_id={}]", request.method, request.url.path, req_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": req_id},
            headers={"x-request-id": req_id or ""},
        )
