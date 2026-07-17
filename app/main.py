"""FastAPI application factory for the RAG service."""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.chunking import chunk_text
from app.citations import extract_citations
from app.embeddings import Embedder, build_embedder
from app.errors import ProviderError
from app.llm import LLM, build_llm
from app.rerank import Reranker, build_reranker
from app.settings import Settings
from app.store import (
    ChunkRecord,
    DocumentRecord,
    PgVectorStore,
    VectorStore,
    build_store,
    search_with_mode,
)

# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #

# Input bounds. These keep a single request's memory/compute footprint finite
# so hostile payloads (a 2 MB document, a 500-document batch, a 50 KB question)
# are rejected with 422 instead of ballooning the in-memory index. The limits
# are ~100x real usage (corpus docs are ~2.5 KB), so they never bite normal use.
MAX_TITLE_CHARS = 1_000
MAX_TEXT_CHARS = 1_000_000  # 1 MB
MAX_QUESTION_CHARS = 10_000
MAX_DOCS_PER_REQUEST = 100
MAX_METADATA_BYTES = 64 * 1024  # 64 KB of JSON-serialised metadata per document


class DocumentIn(BaseModel):
    id: Optional[str] = None
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    metadata: dict = Field(default_factory=dict)

    @field_validator("title", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length=1 lets whitespace-only strings through; reject them so a
        # document that would index to zero chunks never becomes a ghost record.
        if not value.strip():
            raise ValueError("must not be blank or whitespace-only")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_within_bound(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > MAX_METADATA_BYTES:
            raise ValueError(
                f"metadata too large (limit is {MAX_METADATA_BYTES} bytes serialised)"
            )
        return value


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_DOCS_PER_REQUEST)

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> "IngestRequest":
        # Two documents sharing an explicit id in one batch is ambiguous: the
        # second would silently overwrite the first's chunks while chunks_indexed
        # still counted both. Reject it so ingest stays deterministic.
        explicit_ids = [d.id for d in self.documents if d.id is not None]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("duplicate document ids within a single request are not allowed")
        return self


class IngestResponse(BaseModel):
    document_ids: list[str]
    chunks_indexed: int
    skipped: int  # documents whose content hash was unchanged (not re-indexed)


def content_hash(title: str, text: str) -> str:
    """Stable identity of a document's indexed content (title + text)."""
    return hashlib.sha256(f"{title}\n{text}".encode("utf-8")).hexdigest()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank or whitespace-only")
        return value


class CitationOut(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    snippet: str
    score: float


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    ord: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    usage: Optional[dict] = None


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app(
    settings: Optional[Settings] = None,
    *,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    llm: Optional[LLM] = None,
    reranker: Optional[Reranker] = None,
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

    async def require_api_key(request: Request) -> None:
        """Bearer auth against the RAG_API_KEYS set (401 on any mismatch)."""
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        token = token.strip()
        authorized = scheme.lower() == "bearer" and any(
            secrets.compare_digest(token, key) for key in api_keys
        )
        if not authorized:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid API key (send 'Authorization: Bearer <key>')",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authed = [Depends(require_api_key)]

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        return JSONResponse(
            status_code=502, content={"detail": f"upstream provider error: {exc}"}
        )

    @app.post("/ingest", response_model=IngestResponse, dependencies=authed)
    async def ingest(payload: IngestRequest) -> IngestResponse:
        st: Settings = app.state.settings
        document_ids: list[str] = []
        incoming: list[tuple[DocumentIn, str, str]] = []  # (doc, id, hash)

        for doc in payload.documents:
            doc_id = doc.id or uuid.uuid4().hex
            document_ids.append(doc_id)
            incoming.append((doc, doc_id, content_hash(doc.title, doc.text)))

        # Idempotence: a document whose (title, text) hash matches what is
        # already stored is skipped before any chunking or embedding work.
        existing = await app.state.store.content_hashes([i[1] for i in incoming])
        skipped = 0
        docs_with_chunks: list[tuple[DocumentRecord, list[str]]] = []
        all_texts: list[str] = []
        for doc, doc_id, doc_hash in incoming:
            if existing.get(doc_id) == doc_hash:
                skipped += 1
                continue
            pieces = chunk_text(doc.text, st.chunk_size, st.chunk_overlap)
            record = DocumentRecord(
                id=doc_id, title=doc.title, metadata=doc.metadata, content_hash=doc_hash
            )
            docs_with_chunks.append((record, pieces))
            all_texts.extend(pieces)

        # One batched embedding call for the whole request.
        vectors = await app.state.embedder.embed(all_texts) if all_texts else []

        cursor = 0
        chunks_indexed = 0
        for record, pieces in docs_with_chunks:
            chunk_records = [
                ChunkRecord(
                    id=f"{record.id}:{i}",
                    document_id=record.id,
                    ord=i,
                    content=piece,
                    embedding=vectors[cursor + i],
                )
                for i, piece in enumerate(pieces)
            ]
            cursor += len(pieces)
            await app.state.store.upsert(record, chunk_records)
            chunks_indexed += len(chunk_records)

        return IngestResponse(
            document_ids=document_ids, chunks_indexed=chunks_indexed, skipped=skipped
        )

    @app.post("/query", response_model=QueryResponse, dependencies=authed)
    async def query(payload: QueryRequest) -> QueryResponse:
        st: Settings = app.state.settings
        query_vec = (await app.state.embedder.embed([payload.question]))[0]
        retrieved = await search_with_mode(
            app.state.store, st.search_mode, query_vec, payload.question, payload.top_k
        )
        if app.state.reranker is not None:
            retrieved = await app.state.reranker.rerank(payload.question, retrieved)
        result = await app.state.llm.answer(payload.question, retrieved)
        citations = extract_citations(result.answer, retrieved)
        return QueryResponse(
            answer=result.answer,
            citations=[CitationOut(**c.__dict__) for c in citations],
            retrieved=[
                RetrievedChunkOut(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    ord=c.ord,
                    score=c.score,
                )
                for c in retrieved
            ],
            usage=result.usage,
        )

    @app.get("/stats", dependencies=authed)
    async def stats() -> dict:
        st: Settings = app.state.settings
        store_stats = await app.state.store.stats()
        return {
            **store_stats,
            "embeddings_backend": st.embeddings_backend,
            "llm_backend": st.llm_backend,
            "embedding_dim": app.state.embedder.dim,
            "search_mode": st.search_mode,
            "reranker": app.state.reranker.name if app.state.reranker else "none",
        }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app
