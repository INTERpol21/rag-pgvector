"""Shared exception types and the app's exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ProviderError(RuntimeError):
    """An upstream provider (embeddings API or LLM) failed.

    Raised on network errors, timeouts and non-2xx responses from
    OpenAI-compatible backends. The API layer maps it to HTTP 502.
    """


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content={"detail": f"upstream provider error: {exc}"}
        )
