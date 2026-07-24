"""Bearer-token authentication dependency (constant-time comparison)."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


async def require_api_key(request: Request) -> None:
    """Bearer auth against the RAG_API_KEYS set (401 on any mismatch)."""
    api_keys: frozenset[str] = request.app.state.api_keys
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    token = token.strip()
    # Compare against EVERY key, never short-circuiting: ``any()`` returns on
    # the first match, so validation time would leak which key matched —
    # the same strict contract the gateway and orchestrator already keep.
    matched = False
    for key in api_keys:
        if secrets.compare_digest(token, key):
            matched = True
    authorized = scheme.lower() == "bearer" and matched
    if not authorized:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key (send 'Authorization: Bearer <key>')",
            headers={"WWW-Authenticate": "Bearer"},
        )
