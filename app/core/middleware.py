"""HTTP middleware: request-id propagation and one structured log line per request."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

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


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Honor an incoming ``X-Request-ID`` (or mint one), echo it, log one line."""
    request_id = _sanitize_request_id(request.headers.get("x-request-id"))
    # Expose it on request.state so exception handlers can correlate their logs.
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
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
