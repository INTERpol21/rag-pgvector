import os

import pytest

from app.store import (
    SEARCH_SQL,
    ChunkRecord,
    DocumentRecord,
    PgVectorStore,
    normalize_dsn,
)


def test_dsn_normalization():
    assert (
        normalize_dsn("postgresql+asyncpg://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db"
    )
    assert normalize_dsn("postgres://u@h/db") == "postgresql://u@h/db"
    assert normalize_dsn(" postgresql://u@h/db ") == "postgresql://u@h/db"


def test_schema_sql_and_lazy_construction():
    store = PgVectorStore("postgresql+asyncpg://u:p@h/db", dim=256)
    assert store.dsn == "postgresql://u:p@h/db"

    sql = store.schema_sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "vector(256)" in sql
    assert "ON DELETE CASCADE" in sql
    assert "hnsw" in sql  # ANN index (migrations/002_hnsw.sql) is part of the schema

    assert "<=>" in SEARCH_SQL and "ORDER BY" in SEARCH_SQL

    # constructor must be side-effect free; use before connect() fails loudly
    with pytest.raises(RuntimeError, match="not connected"):
        store._require_pool()


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="integration test requires a live Postgres with pgvector (set DATABASE_URL)",
)
async def test_pgvector_roundtrip_integration():
    store = PgVectorStore(os.environ["DATABASE_URL"], dim=4)
    await store.connect()
    try:
        await store.ensure_schema()
        await store.upsert(
            DocumentRecord(id="it-doc", title="Integration Doc", metadata={"k": "v"}),
            [
                ChunkRecord("it-doc:0", "it-doc", 0, "first", [1.0, 0.0, 0.0, 0.0]),
                ChunkRecord("it-doc:1", "it-doc", 1, "second", [0.0, 1.0, 0.0, 0.0]),
            ],
        )
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert results[0].chunk_id == "it-doc:0"
        assert results[0].score > results[1].score
        stats = await store.stats()
        assert stats["documents"] >= 1 and stats["chunks"] >= 2
    finally:
        await store.close()
