"""Vector store backends behind a minimal ``VectorStore`` protocol.

Two implementations:

* ``MemoryVectorStore`` — pure-python cosine search; zero dependencies,
  used by tests, evals and the offline quickstart.
* ``PgVectorStore`` — Postgres + pgvector via asyncpg, cosine ``<=>``
  ordered search; used in docker-compose / production mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    title: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    ord: int
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    ord: int
    score: float  # cosine similarity in [-1, 1], higher is better


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_schema(self) -> None: ...

    async def upsert(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None: ...

    async def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    async def stats(self) -> dict: ...


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryVectorStore:
    """Dict-backed store with exact (brute-force) cosine search."""

    backend = "memory"

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, ChunkRecord] = {}

    async def ensure_schema(self) -> None:  # nothing to do for dicts
        return None

    async def upsert(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None:
        # Re-ingesting a document replaces its chunks (FK cascade semantics).
        self._chunks = {
            cid: c for cid, c in self._chunks.items() if c.document_id != document.id
        }
        self._documents[document.id] = document
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    async def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                title=self._documents[c.document_id].title,
                content=c.content,
                ord=c.ord,
                score=_cosine(embedding, c.embedding),
            )
            for c in self._chunks.values()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: max(top_k, 0)]

    async def stats(self) -> dict:
        return {
            "backend": self.backend,
            "documents": len(self._documents),
            "chunks": len(self._chunks),
        }


# --------------------------------------------------------------------------- #
# Postgres + pgvector backend
# --------------------------------------------------------------------------- #

SCHEMA_SQL_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id         text PRIMARY KEY,
    title      text NOT NULL,
    metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         int  NOT NULL,
    content     text NOT NULL,
    embedding   vector({dim}) NOT NULL
);

-- At scale, add an ANN index and trade a little recall for a lot of speed:
--   CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat
--       ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- (build it AFTER bulk-loading data; tune `lists` ~ sqrt(rows), and
--  `SET ivfflat.probes` per session for the recall/latency trade-off).
"""

SEARCH_SQL = """
SELECT c.id, c.document_id, d.title, c.content, c.ord,
       1 - (c.embedding <=> $1) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> $1
LIMIT $2
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (id, title, metadata)
VALUES ($1, $2, $3::jsonb)
ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, metadata = EXCLUDED.metadata
"""

DELETE_CHUNKS_SQL = "DELETE FROM chunks WHERE document_id = $1"

INSERT_CHUNK_SQL = """
INSERT INTO chunks (id, document_id, ord, content, embedding)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (id) DO UPDATE
    SET document_id = EXCLUDED.document_id, ord = EXCLUDED.ord,
        content = EXCLUDED.content, embedding = EXCLUDED.embedding
"""


def normalize_dsn(dsn: str) -> str:
    """Normalize SQLAlchemy-style DSNs to plain libpq form for asyncpg."""
    dsn = dsn.strip()
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return dsn


class PgVectorStore:
    """pgvector-backed store (asyncpg pool, cosine ``<=>`` search).

    The constructor is side-effect free; call :meth:`connect` (done by the
    app lifespan) before use.
    """

    backend = "pgvector"

    def __init__(self, dsn: str, dim: int) -> None:
        self.dsn = normalize_dsn(dsn)
        self.dim = dim
        self._pool = None

    @property
    def schema_sql(self) -> str:
        return SCHEMA_SQL_TEMPLATE.format(dim=self.dim)

    async def connect(self) -> None:
        import asyncpg
        from pgvector.asyncpg import register_vector

        async def _init(conn) -> None:
            # The extension must exist before the vector codec can register.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector(conn)

        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=1, max_size=5, init=_init
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PgVectorStore is not connected; call connect() first")
        return self._pool

    async def ensure_schema(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(self.schema_sql)

    async def upsert(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None:
        import json

        from pgvector import Vector

        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    UPSERT_DOCUMENT_SQL,
                    document.id,
                    document.title,
                    json.dumps(document.metadata),
                )
                await conn.execute(DELETE_CHUNKS_SQL, document.id)
                await conn.executemany(
                    INSERT_CHUNK_SQL,
                    [
                        (c.id, c.document_id, c.ord, c.content, Vector(c.embedding))
                        for c in chunks
                    ],
                )

    async def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]:
        from pgvector import Vector

        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SEARCH_SQL, Vector(list(embedding)), top_k)
        return [
            ScoredChunk(
                chunk_id=r["id"],
                document_id=r["document_id"],
                title=r["title"],
                content=r["content"],
                ord=r["ord"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def stats(self) -> dict:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            documents = await conn.fetchval("SELECT count(*) FROM documents")
            chunks = await conn.fetchval("SELECT count(*) FROM chunks")
        return {"backend": self.backend, "documents": documents, "chunks": chunks}


def build_store(settings, dim: int) -> VectorStore:
    """Instantiate the store selected by ``STORE_BACKEND``."""
    backend = settings.store_backend.lower()
    if backend == "memory":
        return MemoryVectorStore()
    if backend == "pgvector":
        return PgVectorStore(dsn=settings.database_url, dim=dim)
    raise ValueError(f"unknown STORE_BACKEND: {settings.store_backend!r}")
