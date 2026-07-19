"""Deep integration tests against a REAL Postgres + pgvector (gated).

Every test here takes the ``fresh_pg_dsn`` fixture, which hands out a private,
empty database, so exact document/chunk counts and a from-scratch migration run
are deterministic. These target the costly real-DB failure modes — a migration
that doesn't apply, silent chunk accumulation on re-ingest, local-first ranking
that only works in memory, a dimension change that corrupts inserts, ungrounded
citations — not glue that the offline suite already covers.

Set ``DATABASE_URL`` (e.g. postgresql://rag:rag@localhost:5432/ragdeep) to run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest

from app.core.settings import Settings
from app.db.store import (
    ChunkRecord,
    DocumentRecord,
    PgVectorStore,
    search_with_mode,
)
from app.main import create_app
from app.services.embeddings import HashingEmbedder
from app.services.llm import NOT_IN_SOURCES_ANSWER

AUTH_HEADERS = {"Authorization": "Bearer demo-key"}
DIM = 4  # integration convention (pgvector 0.6 here)


@asynccontextmanager
async def pg_app(dsn: str, **overrides):
    """A create_app() client wired to a connected PgVectorStore.

    httpx's ASGITransport doesn't run the app lifespan, so we connect and
    migrate the store by hand (exactly what the lifespan does in production)
    and tear it down afterwards. This drives the full HTTP pipeline
    (chunk -> embed -> upsert / retrieve -> synthesize) against real Postgres.
    """
    settings = Settings(
        store_backend="pgvector",
        embeddings_backend="hash",
        embedding_dim=DIM,
        llm_backend="grounded",
        search_mode="hybrid",
        rag_api_keys="demo-key",
        **overrides,
    )
    store = PgVectorStore(dsn, dim=DIM)
    embedder = HashingEmbedder(dim=DIM)
    await store.connect()
    await store.ensure_schema()
    app = create_app(settings, store=store, embedder=embedder)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers=AUTH_HEADERS
        ) as client:
            yield client, store
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# Migrations apply cleanly on a fresh database
# --------------------------------------------------------------------------- #


async def test_migrations_apply_cleanly_on_fresh_db(fresh_pg_dsn):
    store = PgVectorStore(fresh_pg_dsn, dim=DIM)
    await store.connect()
    try:
        await store.ensure_schema()
        pool = store._require_pool()
        async with pool.acquire() as conn:
            versions = {
                r["version"]
                for r in await conn.fetch("SELECT version FROM schema_migrations")
            }
            indexes = {
                r["indexname"]: r["indexdef"]
                for r in await conn.fetch(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE tablename IN ('chunks', 'documents')"
                )
            }
            doc_cols = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'documents'"
                )
            }
            chunk_cols = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks'"
                )
            }
            n_applied = await conn.fetchval("SELECT count(*) FROM schema_migrations")

        # Every migration file recorded (001..006 at least).
        for stem in ("001_init", "002_hnsw", "003_fts", "004_document_id_index",
                     "005_source_tags", "006_hnsw_tuning"):
            assert stem in versions, versions

        # 003: generated tsvector column + GIN index for the keyword leg.
        assert "content_tsv" in chunk_cols
        assert "chunks_content_tsv_gin" in indexes

        # 004: FK index used by re-ingest DELETE and cascade delete.
        assert "chunks_document_id_idx" in indexes

        # 005: local-first provenance columns + source filter index.
        assert {"source", "priority", "owner"} <= doc_cols
        assert "documents_source_idx" in indexes

        # 006: HNSW recreated with the tuned build parameters.
        hnsw = indexes.get("chunks_embedding_hnsw", "")
        assert "hnsw" in hnsw
        assert "m='16'" in hnsw and "ef_construction='200'" in hnsw, hnsw

        # Re-running is idempotent: no new migration rows, no error.
        await store.ensure_schema()
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM schema_migrations") == n_applied
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# Scale ingest -> hybrid top-k + re-ingest idempotence
# --------------------------------------------------------------------------- #


def _scale_docs(n: int) -> list[dict]:
    # Each doc carries a globally unique rare token so the FTS leg of hybrid
    # search can pinpoint it; shared filler keeps the corpus realistic.
    docs = []
    for k in range(n):
        docs.append(
            {
                "id": f"doc{k:03d}",
                "title": f"Note {k}",
                "text": (
                    f"needletoken{k:03d} is the unique marker for note {k}. "
                    "This note also discusses vector search and retrieval in general."
                ),
            }
        )
    return docs


async def test_scale_ingest_hybrid_topk_and_reingest_idempotent(fresh_pg_dsn):
    n = 30
    docs = _scale_docs(n)
    async with pg_app(fresh_pg_dsn) as (client, store):
        resp = await client.post("/ingest", json={"documents": docs})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chunks_indexed"] == n and body["skipped"] == 0

        stats = await store.stats()
        assert stats["documents"] == n
        chunk_total = stats["chunks"]
        assert chunk_total == n  # each short doc -> exactly one chunk

        # Hybrid search returns the right doc on top for several targets.
        for k in (0, 7, 15, 29):
            q = f"needletoken{k:03d}"
            r = (await client.post("/query", json={"question": q, "top_k": 5})).json()
            assert r["retrieved"], r
            assert len(r["retrieved"]) <= 5
            assert r["retrieved"][0]["document_id"] == f"doc{k:03d}", (k, r["retrieved"][0])

        # Re-ingest the identical batch: content-hash idempotence skips them all,
        # and the chunk count is unchanged (no silent accumulation).
        resp = await client.post("/ingest", json={"documents": docs})
        body = resp.json()
        assert body["skipped"] == n and body["chunks_indexed"] == 0
        stats2 = await store.stats()
        assert stats2 == stats

        # Re-ingest one id with NEW content: chunks are replaced, not added.
        changed = {"id": "doc001", "title": "Note 1", "text":
                   "needletoken001 rewritten. Fresh body about hybrid ranking and fusion."}
        resp = await client.post("/ingest", json={"documents": [changed]})
        assert resp.json()["skipped"] == 0
        stats3 = await store.stats()
        assert stats3["documents"] == n and stats3["chunks"] == chunk_total
        # The rewritten doc is still retrievable by its marker.
        r = (await client.post("/query", json={"question": "needletoken001", "top_k": 3})).json()
        assert r["retrieved"][0]["document_id"] == "doc001"


# --------------------------------------------------------------------------- #
# Local-first ranking on real pgvector (extended)
# --------------------------------------------------------------------------- #


async def test_local_first_ranking_and_strict_filter_extended(fresh_pg_dsn):
    # Six docs with IDENTICAL text (so raw similarity ties) but different
    # provenance: 3 local (priority 100) vs 3 web (priority 0), plus one 'other'.
    shared = "hybrid retrieval fuses vector and keyword ranking for pgvector."
    docs = []
    for k in range(3):
        docs.append({"id": f"loc{k}", "title": f"Local {k}", "text": shared,
                     "source": "local", "priority": 100})
        docs.append({"id": f"web{k}", "title": f"Web {k}", "text": shared,
                     "source": "web", "priority": 0})
    docs.append({"id": "oth0", "title": "Other", "text": shared,
                 "source": "other", "priority": 50})

    async with pg_app(fresh_pg_dsn) as (client, _store):
        assert (await client.post("/ingest", json={"documents": docs})).status_code == 200
        q = "hybrid retrieval vector keyword ranking pgvector"

        # No filter: local (highest priority) must occupy the top slots, web last.
        body = (await client.post("/query", json={"question": q, "top_k": 7})).json()
        sources = [c["source"] for c in body["retrieved"]]
        assert sources, body
        # every local appears before every web (priority boost, ties by score/id)
        local_positions = [i for i, s in enumerate(sources) if s == "local"]
        web_positions = [i for i, s in enumerate(sources) if s == "web"]
        assert local_positions and web_positions
        assert max(local_positions) < min(web_positions)
        # the 'other' tier (priority 50) sits between local and web
        other_positions = [i for i, s in enumerate(sources) if s == "other"]
        if other_positions:
            assert max(local_positions) < other_positions[0] < min(web_positions)

        # top_k=3 with 3 local available -> returns only local (boost fills first).
        body = (await client.post("/query", json={"question": q, "top_k": 3})).json()
        assert [c["source"] for c in body["retrieved"]] == ["local", "local", "local"]

        # Strict sources=["local"] excludes web AND other across the whole corpus.
        body = (await client.post(
            "/query", json={"question": q, "top_k": 7, "sources": ["local"]}
        )).json()
        assert body["retrieved"]
        assert all(c["source"] == "local" for c in body["retrieved"])
        assert {c["document_id"] for c in body["retrieved"]} == {"loc0", "loc1", "loc2"}

        # Mixed filter ["local","other"] keeps both tiers, still excludes web.
        body = (await client.post(
            "/query", json={"question": q, "top_k": 7, "sources": ["local", "other"]}
        )).json()
        got = {c["source"] for c in body["retrieved"]}
        assert got <= {"local", "other"} and "web" not in got
        assert "other" in got


# --------------------------------------------------------------------------- #
# Startup dimension guard on a fresh database
# --------------------------------------------------------------------------- #


async def test_dimension_guard_rejects_reconnect_with_new_dim(fresh_pg_dsn):
    at_four = PgVectorStore(fresh_pg_dsn, dim=DIM)
    await at_four.connect()
    try:
        await at_four.ensure_schema()  # column declared vector(4)
    finally:
        await at_four.close()

    # Reconnecting the SAME database with a different embedding dim must fail at
    # ensure_schema() (startup), not silently on every later insert.
    at_eight = PgVectorStore(fresh_pg_dsn, dim=DIM + 4)
    await at_eight.connect()
    try:
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            await at_eight.ensure_schema()
    finally:
        await at_eight.close()


# --------------------------------------------------------------------------- #
# Grounding / citation honesty against real retrieval
# --------------------------------------------------------------------------- #


GROUNDING_CORPUS = [
    {"id": "pg", "title": "pgvector", "text":
        "pgvector adds a vector column to Postgres. Cosine distance uses the "
        "operator vector_cosine_ops for approximate nearest neighbor search."},
    {"id": "coffee", "title": "Espresso", "text":
        "Espresso extraction depends on grind size dose and water temperature. "
        "A typical shot uses eighteen grams and a short pour."},
]


async def test_grounded_query_cites_only_retrieved_on_real_db(fresh_pg_dsn):
    async with pg_app(fresh_pg_dsn) as (client, _store):
        assert (await client.post(
            "/ingest", json={"documents": GROUNDING_CORPUS}
        )).status_code == 200

        body = (await client.post(
            "/query",
            json={"question": "Which opclass does pgvector use for cosine distance?", "top_k": 4},
        )).json()
        assert body["answer"] != NOT_IN_SOURCES_ANSWER
        assert body["citations"], body
        retrieved_ids = {r["chunk_id"] for r in body["retrieved"]}
        for c in body["citations"]:
            assert c["chunk_id"] in retrieved_ids  # never cite outside retrieval

        # Off-topic question: real retrieval still returns rows, grounded LLM abstains.
        body = (await client.post(
            "/query", json={"question": "Who won the world cup in 1998?", "top_k": 4}
        )).json()
        assert body["answer"] == NOT_IN_SOURCES_ANSWER
        assert body["citations"] == []


# --------------------------------------------------------------------------- #
# Direct store-level local-first (mirrors the existing shared-DB test, isolated)
# --------------------------------------------------------------------------- #


async def test_store_level_priority_boost_at_equal_similarity(fresh_pg_dsn):
    store = PgVectorStore(fresh_pg_dsn, dim=DIM)
    await store.connect()
    try:
        await store.ensure_schema()
        emb = [0.0, 1.0, 0.0, 0.0]
        await store.upsert(
            DocumentRecord(id="w", title="Web", source="web", priority=0),
            [ChunkRecord("w:0", "w", 0, "vector cosine distance", emb)],
        )
        await store.upsert(
            DocumentRecord(id="l", title="Local", source="local", priority=100),
            [ChunkRecord("l:0", "l", 0, "vector cosine distance", emb)],
        )
        boosted = await search_with_mode(store, "vector", emb, "vector cosine", 10)
        assert boosted[0].chunk_id == "l:0"  # local wins at equal similarity
        strict = await search_with_mode(
            store, "vector", emb, "vector cosine", 10, sources={"local"}
        )
        assert [c.chunk_id for c in strict] == ["l:0"]
    finally:
        await store.close()
