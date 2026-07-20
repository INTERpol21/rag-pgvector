"""Shared exception types and the app's exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger("rag.errors")


class ProviderError(RuntimeError):
    """An upstream provider (embeddings API or LLM) failed.

    Raised on network errors, timeouts and non-2xx responses from
    OpenAI-compatible backends. The API layer maps it to HTTP 502.
    """


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
        # Log the full cause server-side (it can carry the upstream URL or a raw
        # upstream response body), but return only a generic detail to the client
        # so internal topology / upstream content never leaks across the API.
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "upstream provider error: %s", exc, extra={"request_id": request_id}
        )
        return JSONResponse(
            status_code=502, content={"detail": "upstream provider unavailable"}
        )
