"""HTTP middleware: request-id propagation and one structured log line per request."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger("rag")

# Cap a client-supplied X-Request-ID: it is echoed back and logged, so bound its
# length to keep a hostile client from bloating headers/log lines.
_MAX_REQUEST_ID_LEN = 128


def _sanitize_request_id(raw: str | None) -> str:
    """Return a bounded request id: the caller's (trimmed) or a fresh one."""
    if raw:
        candidate = raw.strip()[:_MAX_REQUEST_ID_LEN]
        if candidate:
            return candidate
    return uuid.uuid4().hex[:16]


def oversize_response(request: Request) -> JSONResponse | None:
    """413 when the client declares a body larger than the configured cap.

    Same contract as the gateway's: rejecting on ``Content-Length`` stops an
    oversized payload before FastAPI buffers and pydantic parses it (the
    schema bounds alone allowed ~100 MB of JSON into memory first). Chunked
    requests without the header pass through to the schema limits.
    """
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    try:
        size = int(declared)
    except ValueError:
        return None
    limit: int = request.app.state.settings.max_request_bytes
    if size <= limit:
        return None
    return JSONResponse(
        {"detail": f"request body is {size} bytes, over the {limit}-byte limit"},
        status_code=413,
    )


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Honor an incoming ``X-Request-ID`` (or mint one), echo it, log one line."""
    request_id = _sanitize_request_id(request.headers.get("x-request-id"))
    # Expose it on request.state so exception handlers can correlate their logs.
    request.state.request_id = request_id
    started = time.perf_counter()
    oversized = oversize_response(request)
    response: Response = oversized if oversized is not None else await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers.setdefault("X-Request-ID", request_id)
    logger.info(
        "%s %s -> %d in %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        extra={"request_id": request_id},
    )
    return response
