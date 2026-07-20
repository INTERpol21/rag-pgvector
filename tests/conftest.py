import os
import uuid

import httpx
import pytest

from app.core.settings import Settings
from app.db.store import MemoryVectorStore, normalize_dsn
from app.main import create_app
from app.services.embeddings import HashingEmbedder
from app.services.llm import MockLLM

# The API requires a bearer key (RAG_API_KEYS); every test client sends the
# default one. Auth-specific tests build their own clients without it.
API_KEY = "demo-key"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embeddings_backend="hash",
        store_backend="memory",
        llm_backend="mock",
        rag_api_keys=API_KEY,
    )


@pytest.fixture
def app(settings, store, embedder):
    return create_app(settings, store=store, embedder=embedder, llm=MockLLM())


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as c:
        yield c


DOCS = [
    {
        "id": "gateway",
        "title": "Gateway Guide",
        "text": (
            "An LLM gateway centralizes authentication, rate limiting and cost "
            "tracking. A token bucket limiter refills at a fixed rate and "
            "rejects requests with HTTP 429 when the bucket is empty. "
            "Streaming uses Server-Sent Events."
        ),
    },
    {
        "id": "pgvector",
        "title": "pgvector Guide",
        "text": (
            "pgvector adds a vector column type to Postgres. Cosine distance "
            "uses the <=> operator with the vector_cosine_ops opclass. IVFFlat "
            "and HNSW indexes provide approximate nearest neighbor search; "
            "tune lists and probes for recall."
        ),
    },
    {
        "id": "espresso",
        "title": "Espresso Handbook",
        "text": (
            "Espresso extraction depends on grind size, dose and water "
            "temperature. A typical shot uses eighteen grams of coffee, a "
            "twenty five second pour and ninety three degree water. Crema "
            "signals fresh beans."
        ),
    },
]


@pytest.fixture
async def ingested_client(client):
    resp = await client.post("/v1/ingest", json={"documents": DOCS})
    assert resp.status_code == 200, resp.text
    return client


# --------------------------------------------------------------------------- #
# Integration (real pgvector) helpers
# --------------------------------------------------------------------------- #


def _swap_database(dsn: str, name: str) -> str:
    """Return ``dsn`` pointed at database ``name`` (drops any query string)."""
    head = dsn.rsplit("/", 1)[0]
    return f"{head}/{name}"


@pytest.fixture
async def fresh_pg_dsn():
    """Provision an isolated, empty pgvector database for one test.

    Gated on ``DATABASE_URL``. A uniquely named database is created on the same
    server, the ``vector`` extension installed, its DSN yielded, then the
    database is dropped on teardown. Isolation lets count-sensitive integration
    tests (exact document/chunk totals, a from-scratch migration run) assert on a
    known-empty schema without colliding with the shared roundtrip/local-first
    rows other tests leave in the primary database.
    """
    base = os.getenv("DATABASE_URL")
    if not base:
        pytest.skip("integration test requires a live Postgres with pgvector (set DATABASE_URL)")
    import asyncpg

    base = normalize_dsn(base)
    name = f"ragdeep_{uuid.uuid4().hex[:12]}"

    admin = await asyncpg.connect(base)
    try:
        await admin.execute(f'CREATE DATABASE "{name}" OWNER rag')
    finally:
        await admin.close()

    dsn = _swap_database(base, name)
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()

    try:
        yield dsn
    finally:
        admin = await asyncpg.connect(base)
        try:
            # Force-close any lingering pool connections so DROP can proceed.
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()
