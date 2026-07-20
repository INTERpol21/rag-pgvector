"""Shared FastAPI dependencies for the HTTP routes."""

from app.core.security import require_api_key

__all__ = ["require_api_key"]
