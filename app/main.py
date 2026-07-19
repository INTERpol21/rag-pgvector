"""FastAPI application factory for the RAG service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, ingest, query, stats
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import request_context
from app.core.settings import Settings
from app.db.store import PgVectorStore, VectorStore, build_store
from app.services.embeddings import Embedder, build_embedder
from app.services.llm import LLM, build_llm
from app.services.rerank import Reranker, build_reranker


def create_app(
    settings: Settings | None = None,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Build the application.

    Components not supplied explicitly (as tests do) are constructed from
    settings, which in turn default to the fully offline stack
    (memory store + hashing embedder + mock LLM, no reranker).
    """
    settings = settings or Settings()
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings, dim=embedder.dim)
    llm = llm or build_llm(settings)
    reranker = reranker or build_reranker(settings, llm)
    api_keys = frozenset(
        key.strip() for key in settings.rag_api_keys.split(",") if key.strip()
    )
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Only the pgvector backend owns real resources (asyncpg pool).
        if isinstance(app.state.store, PgVectorStore):
            await app.state.store.connect()
            await app.state.store.ensure_schema()
            try:
                yield
            finally:
                await app.state.store.close()
        else:
            await app.state.store.ensure_schema()
            yield

    app = FastAPI(
        title="rag-pgvector",
        version="0.1.0",
        description="RAG service: ingest -> chunk -> embed -> vector search -> "
        "LLM synthesis with [n] citations.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.reranker = reranker
    app.state.api_keys = api_keys

    app.middleware("http")(request_context)
    register_exception_handlers(app)
    for route_module in (health, ingest, query, stats):
        app.include_router(route_module.router)
    return app
