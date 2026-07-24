"""FastAPI application factory for the RAG service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import __version__
from app.api.routes import health, ingest, query, stats
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import request_context
from app.core.settings import Settings
from app.db.store import PgVectorStore, VectorStore, build_store
from app.services.embeddings import Embedder, build_embedder
from app.services.llm import LLM, build_llm
from app.services.reindex import sync_embedder_fingerprint
from app.services.rerank import Reranker, build_reranker
from app.services.watcher import watch_forever


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
    logger = get_logger("rag.app")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Only the pgvector backend owns real resources (asyncpg pool).
        if isinstance(app.state.store, PgVectorStore):
            await app.state.store.connect()
            # ensure_schema inside the try: its dimension guard is a documented
            # fail-fast path, and failing it must still close the asyncpg pool.
            # The fingerprint sync runs before traffic for the same reason: a
            # switched embedder re-embeds the corpus (or the startup fails)
            # rather than serving mixed vector spaces.
            try:
                await app.state.store.ensure_schema()
                await _sync_fingerprint(app)
                watcher = _start_watcher(app)
                try:
                    yield
                finally:
                    await _stop_watcher(watcher)
            finally:
                await app.state.store.close()
                await _close_clients(app)
        else:
            await app.state.store.ensure_schema()
            await _sync_fingerprint(app)
            watcher = _start_watcher(app)
            try:
                yield
            finally:
                await _stop_watcher(watcher)
                await _close_clients(app)

    def _start_watcher(app: FastAPI) -> asyncio.Task[None] | None:
        if not settings.ingest_watch_dir:
            return None
        root = Path(settings.ingest_watch_dir)
        root.mkdir(parents=True, exist_ok=True)
        return asyncio.create_task(
            watch_forever(
                root,
                store=app.state.store,
                embedder=app.state.embedder,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                interval_s=settings.ingest_watch_interval_s,
                logger=logger,
            )
        )

    async def _stop_watcher(task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _close_clients(app: FastAPI) -> None:
        # The OpenAI-backed embedder/LLM hold a shared httpx client (keep-alive
        # to the gateway); offline mocks have no aclose — duck-typed on purpose
        # so the Protocols stay minimal.
        for component in (app.state.embedder, app.state.llm):
            aclose = getattr(component, "aclose", None)
            if aclose is not None:
                await aclose()

    async def _sync_fingerprint(app: FastAPI) -> None:
        outcome = await sync_embedder_fingerprint(app.state.store, app.state.embedder)
        logger.info(
            "embedder fingerprint %s (%s)", outcome, app.state.embedder.fingerprint
        )

    app = FastAPI(
        title="rag-pgvector",
        version=__version__,
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
    # Liveness stays unversioned; the API is served under a single /v1 prefix
    # (unified with the gateway so the contracts package has one shape).
    app.include_router(health.router)
    for route_module in (ingest, query, stats):
        app.include_router(route_module.router, prefix="/v1")
    return app
