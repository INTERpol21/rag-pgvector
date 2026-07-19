"""Vector store backends behind a minimal ``VectorStore`` protocol.

Two implementations:

* ``MemoryVectorStore`` — pure-python cosine search plus a compact BM25 over
  a maintained token index; zero dependencies, used by tests, evals and the
  offline quickstart.
* ``PgVectorStore`` — Postgres + pgvector via asyncpg, cosine ``<=>`` ordered
  search plus ``tsvector`` full-text search; used in docker-compose /
  production mode. Schema is applied from ``migrations/*.sql``.

Both expose ``search`` (vector-only) and ``search_hybrid`` (vector + keyword
rankings merged with Reciprocal Rank Fusion). ``SEARCH_MODE`` picks one via
:func:`search_with_mode`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from app.migrations import (
    CREATE_MIGRATIONS_TABLE_SQL,
    IS_APPLIED_SQL,
    RECORD_APPLIED_SQL,
    load_migrations,
)

# Same token model as the hashing embedder: lowercased ASCII word tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 (Okapi) constants: k1 saturates term frequency, b scales length
# normalization. RRF_K dampens the influence of exact ranks in the fusion.
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    title: str
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""  # sha256 over title+text; "" = unknown (pre-hash rows)


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
    score: float  # cosine similarity, BM25 or RRF score depending on the path


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_schema(self) -> None: ...

    async def upsert(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None: ...

    async def content_hashes(self, document_ids: Sequence[str]) -> dict[str, str]: ...

    async def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    async def search_hybrid(
        self, embedding: Sequence[float], query_text: str, top_k: int
    ) -> list[ScoredChunk]: ...

    async def stats(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Rank fusion (shared by both backends)
# --------------------------------------------------------------------------- #


def candidate_pool(top_k: int) -> int:
    """How many candidates each retrieval leg contributes before fusion."""
    return max(top_k * 4, 20)


def rrf_merge(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(id) = sum over lists of 1 / (k + rank).

    Ranks are 1-based; ids absent from a list simply contribute nothing for
    it. Higher is better.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def merge_ranked(
    vector_hits: Sequence[ScoredChunk],
    keyword_hits: Sequence[ScoredChunk],
    top_k: int,
    k: int = RRF_K,
) -> list[ScoredChunk]:
    """RRF-merge two ranked candidate lists into the final top-k.

    The returned chunks carry the RRF score (not cosine/BM25); ties break by
    chunk id so the ordering is deterministic.
    """
    by_id: dict[str, ScoredChunk] = {}
    for chunk in [*vector_hits, *keyword_hits]:
        by_id.setdefault(chunk.chunk_id, chunk)
    scores = rrf_merge(
        [[c.chunk_id for c in vector_hits], [c.chunk_id for c in keyword_hits]], k=k
    )
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        replace(by_id[chunk_id], score=score)
        for chunk_id, score in ordered[: max(top_k, 0)]
    ]


async def search_with_mode(
    store: VectorStore,
    mode: str,
    embedding: Sequence[float],
    query_text: str,
    top_k: int,
) -> list[ScoredChunk]:
    """Dispatch retrieval on ``SEARCH_MODE`` (``vector`` | ``hybrid``)."""
    if mode == "hybrid":
        return await store.search_hybrid(embedding, query_text, top_k)
    return await store.search(embedding, top_k)


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryVectorStore:
    """Dict-backed store: exact cosine search + BM25 over a token index."""

    backend = "memory"

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, ChunkRecord] = {}
        # Token index maintained on upsert: per-chunk term frequencies plus
        # corpus-level document frequencies, so BM25 is O(candidates) a query.
        self._term_freqs: dict[str, Counter] = {}  # chunk id -> token counts
        self._chunk_lens: dict[str, int] = {}  # chunk id -> token count
        self._doc_freq: Counter = Counter()  # token -> chunks containing it

    async def ensure_schema(self) -> None:  # nothing to do for dicts
        return None

    def _unindex_chunk(self, chunk_id: str) -> None:
        for token in self._term_freqs.pop(chunk_id, {}):
            self._doc_freq[token] -= 1
            if self._doc_freq[token] <= 0:
                del self._doc_freq[token]
        self._chunk_lens.pop(chunk_id, None)

    def _index_chunk(self, chunk: ChunkRecord) -> None:
        tokens = _tokenize(chunk.content)
        counts = Counter(tokens)
        self._term_freqs[chunk.id] = counts
        self._chunk_lens[chunk.id] = len(tokens)
        for token in counts:
            self._doc_freq[token] += 1

    async def upsert(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None:
        # Re-ingesting a document replaces its chunks (FK cascade semantics).
        for cid, chunk in list(self._chunks.items()):
            if chunk.document_id == document.id:
                del self._chunks[cid]
                self._unindex_chunk(cid)
        self._documents[document.id] = document
        for chunk in chunks:
            if chunk.id in self._chunks:  # same id from another document
                self._unindex_chunk(chunk.id)
            self._chunks[chunk.id] = chunk
            self._index_chunk(chunk)

    async def content_hashes(self, document_ids: Sequence[str]) -> dict[str, str]:
        return {
            doc_id: self._documents[doc_id].content_hash
            for doc_id in document_ids
            if doc_id in self._documents and self._documents[doc_id].content_hash
        }

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
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[: max(top_k, 0)]

    async def search_bm25(self, query_text: str, top_k: int) -> list[ScoredChunk]:
        """Okapi BM25 (k1=1.5, b=0.75) over the maintained token index."""
        query_tokens = set(_tokenize(query_text))
        n_chunks = len(self._chunks)
        if not query_tokens or n_chunks == 0:
            return []
        avg_len = sum(self._chunk_lens.values()) / n_chunks
        scored: list[ScoredChunk] = []
        for chunk in self._chunks.values():
            freqs = self._term_freqs[chunk.id]
            length_norm = 1.0 - BM25_B + BM25_B * (self._chunk_lens[chunk.id] / avg_len)
            score = 0.0
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                df = self._doc_freq[token]
                idf = math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))
                score += idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * length_norm)
            if score > 0.0:
                scored.append(
                    ScoredChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        title=self._documents[chunk.document_id].title,
                        content=chunk.content,
                        ord=chunk.ord,
                        score=score,
                    )
                )
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[: max(top_k, 0)]

    async def search_hybrid(
        self, embedding: Sequence[float], query_text: str, top_k: int
    ) -> list[ScoredChunk]:
        pool = candidate_pool(top_k)
        vector_hits = await self.search(embedding, top_k=pool)
        keyword_hits = await self.search_bm25(query_text, top_k=pool)
        return merge_ranked(vector_hits, keyword_hits, top_k)

    async def stats(self) -> dict:
        return {
            "backend": self.backend,
            "documents": len(self._documents),
            "chunks": len(self._chunks),
        }


# --------------------------------------------------------------------------- #
# Postgres + pgvector backend
# --------------------------------------------------------------------------- #

SEARCH_SQL = """
SELECT c.id, c.document_id, d.title, c.content, c.ord,
       1 - (c.embedding <=> $1) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> $1
LIMIT $2
"""

# Keyword leg of hybrid search. 'simple' matches the generated content_tsv
# column from migrations/003_fts.sql (no stemming, exact terms searchable);
# websearch_to_tsquery never raises on user input.
KEYWORD_SEARCH_SQL = """
SELECT c.id, c.document_id, d.title, c.content, c.ord,
       ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', $1)) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.content_tsv @@ websearch_to_tsquery('simple', $1)
ORDER BY score DESC, c.id
LIMIT $2
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (id, title, metadata, content_hash)
VALUES ($1, $2, $3::jsonb, $4)
ON CONFLICT (id) DO UPDATE
    SET title = EXCLUDED.title, metadata = EXCLUDED.metadata,
        content_hash = EXCLUDED.content_hash
"""

DELETE_CHUNKS_SQL = "DELETE FROM chunks WHERE document_id = $1"

INSERT_CHUNK_SQL = """
INSERT INTO chunks (id, document_id, ord, content, embedding)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (id) DO UPDATE
    SET document_id = EXCLUDED.document_id, ord = EXCLUDED.ord,
        content = EXCLUDED.content, embedding = EXCLUDED.embedding
"""

CONTENT_HASHES_SQL = """
SELECT id, content_hash FROM documents WHERE id = ANY($1::text[])
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
    """pgvector-backed store (asyncpg pool, cosine ``<=>`` + FTS search).

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
        """Full DDL: every migration in order, ``{dim}`` substituted."""
        return "\n".join(m.sql_for(self.dim) for m in load_migrations())

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
        """Apply pending migrations, tracked in ``schema_migrations``."""
        pool = self._require_pool()
        migrations = load_migrations()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_MIGRATIONS_TABLE_SQL)
            for migration in migrations:
                if await conn.fetchval(IS_APPLIED_SQL, migration.version):
                    continue
                async with conn.transaction():
                    await conn.execute(migration.sql_for(self.dim))
                    await conn.execute(RECORD_APPLIED_SQL, migration.version)

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
                    document.content_hash,
                )
                await conn.execute(DELETE_CHUNKS_SQL, document.id)
                await conn.executemany(
                    INSERT_CHUNK_SQL,
                    [
                        (c.id, c.document_id, c.ord, c.content, Vector(c.embedding))
                        for c in chunks
                    ],
                )

    async def content_hashes(self, document_ids: Sequence[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(CONTENT_HASHES_SQL, list(document_ids))
        # Empty hashes (rows written before hashing existed) never match.
        return {r["id"]: r["content_hash"] for r in rows if r["content_hash"]}

    @staticmethod
    def _scored(rows) -> list[ScoredChunk]:
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

    async def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]:
        from pgvector import Vector

        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SEARCH_SQL, Vector(list(embedding)), top_k)
        return self._scored(rows)

    async def search_hybrid(
        self, embedding: Sequence[float], query_text: str, top_k: int
    ) -> list[ScoredChunk]:
        from pgvector import Vector

        pool_size = candidate_pool(top_k)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            vector_rows = await conn.fetch(
                SEARCH_SQL, Vector(list(embedding)), pool_size
            )
            keyword_rows = await conn.fetch(KEYWORD_SEARCH_SQL, query_text, pool_size)
        return merge_ranked(self._scored(vector_rows), self._scored(keyword_rows), top_k)

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
