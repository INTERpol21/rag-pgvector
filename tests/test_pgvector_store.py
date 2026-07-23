import os

import pytest

from app.db.store import (
    SEARCH_SQL,
    ChunkRecord,
    DocumentRecord,
    PgVectorStore,
    normalize_dsn,
    search_with_mode,
)

_INTEGRATION = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="integration test requires a live Postgres with pgvector (set DATABASE_URL)",
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


@_INTEGRATION
async def test_local_first_boost_and_strict_filter_on_pgvector():
    """Local docs outrank web at equal similarity; strict mode excludes web."""
    store = PgVectorStore(os.environ["DATABASE_URL"], dim=4)
    await store.connect()
    try:
        await store.ensure_schema()
        # A distinct axis from the roundtrip test (which uses [1,0,0,0]) so these
        # rows don't collide in the shared integration DB; identical between the
        # two docs so provenance, not similarity, decides the order.
        emb = [0.0, 0.0, 1.0, 0.0]
        await store.upsert(
            DocumentRecord(id="lf-web", title="Web", source="web", priority=0),
            [ChunkRecord("lf-web:0", "lf-web", 0, "pgvector cosine distance", emb)],
        )
        await store.upsert(
            DocumentRecord(id="lf-loc", title="Local", source="local", priority=100),
            [ChunkRecord("lf-loc:0", "lf-loc", 0, "pgvector cosine distance", emb)],
        )

        boosted = await search_with_mode(store, "vector", emb, "pgvector cosine", 10)
        mine = [c for c in boosted if c.chunk_id in ("lf-web:0", "lf-loc:0")]
        assert mine[0].source == "local"  # priority boost wins

        strict = await search_with_mode(
            store, "vector", emb, "pgvector cosine", 10, sources={"local"}
        )
        assert all(c.source == "local" for c in strict)
        assert any(c.chunk_id == "lf-loc:0" for c in strict)
        assert not any(c.chunk_id == "lf-web:0" for c in strict)
    finally:
        await store.close()


@_INTEGRATION
async def test_dimension_guard_rejects_mismatch():
    """Reconnecting with a different embedding dim fails fast, not on every insert."""
    url = os.environ["DATABASE_URL"]
    probe = PgVectorStore(url, dim=4)
    await probe.connect()
    try:
        pool = probe._require_pool()
        async with pool.acquire() as conn:
            stored = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
            )
        if stored is None:  # fresh DB: create the schema at dim 4
            await probe.ensure_schema()
            stored = 4
    finally:
        await probe.close()

    mismatched = PgVectorStore(url, dim=stored + 7)
    await mismatched.connect()
    try:
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            await mismatched.ensure_schema()
    finally:
        await mismatched.close()


@_INTEGRATION
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


@_INTEGRATION
async def test_russian_word_forms_match_via_fts(fresh_pg_dsn):
    """The pgvector keyword leg must stem Russian: an inflected query has to
    outrank a vector-favoured distractor purely on the FTS hit (migrations/007
    + websearch_to_tsquery('russian')). Under the old 'simple' config the
    keyword leg returns nothing for word forms and the distractor wins."""
    store = PgVectorStore(fresh_pg_dsn, dim=4)
    await store.connect()
    try:
        await store.ensure_schema()
        await store.upsert(
            DocumentRecord(id="ru", title="Заметка"),
            [ChunkRecord("ru:0", "ru", 0, "заметка о векторном поиске", [1.0, 0.0, 0.0, 0.0])],
        )
        await store.upsert(
            DocumentRecord(id="noise", title="Distractor"),
            [ChunkRecord("noise:0", "noise", 0, "nothing relevant here", [0.0, 1.0, 0.0, 0.0])],
        )
        # Query vector favours the distractor; the words favour the Russian
        # chunk only if stemming folds "векторный поиск" onto "векторном поиске".
        results = await store.search_hybrid([0.0, 1.0, 0.0, 0.0], "векторный поиск", top_k=2)
        assert results[0].chunk_id == "ru:0"
    finally:
        await store.close()
