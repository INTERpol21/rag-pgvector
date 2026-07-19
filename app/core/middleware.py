"""HTTP middleware: request-id propagation and one structured log line per request."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from app.core.logging import get_logger

logger = get_logger("rag")


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Honor an incoming ``X-Request-ID`` (or mint one), echo it, log one line."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
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
