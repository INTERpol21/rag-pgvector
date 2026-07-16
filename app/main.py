"""FastAPI application factory for the RAG service."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.citations import extract_citations
from app.embeddings import Embedder, build_embedder
from app.errors import ProviderError
from app.llm import LLM, build_llm
from app.settings import Settings
from app.store import (
    ChunkRecord,
    DocumentRecord,
    PgVectorStore,
    VectorStore,
    build_store,
)

# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #


class DocumentIn(BaseModel):
    id: Optional[str] = None
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class IngestResponse(BaseModel):
    document_ids: list[str]
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


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
) -> FastAPI:
    """Build the application.

    Components not supplied explicitly (as tests do) are constructed from
    settings, which in turn default to the fully offline stack
    (memory store + hashing embedder + mock LLM).
    """
    settings = settings or Settings()
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings, dim=embedder.dim)
    llm = llm or build_llm(settings)

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

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        return JSONResponse(
            status_code=502, content={"detail": f"upstream provider error: {exc}"}
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(payload: IngestRequest) -> IngestResponse:
        st: Settings = app.state.settings
        document_ids: list[str] = []
        docs_with_chunks: list[tuple[DocumentRecord, list[str]]] = []
        all_texts: list[str] = []

        for doc in payload.documents:
            doc_id = doc.id or uuid.uuid4().hex
            pieces = chunk_text(doc.text, st.chunk_size, st.chunk_overlap)
            record = DocumentRecord(id=doc_id, title=doc.title, metadata=doc.metadata)
            docs_with_chunks.append((record, pieces))
            all_texts.extend(pieces)
            document_ids.append(doc_id)

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

        return IngestResponse(document_ids=document_ids, chunks_indexed=chunks_indexed)

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
        query_vec = (await app.state.embedder.embed([payload.question]))[0]
        retrieved = await app.state.store.search(query_vec, top_k=payload.top_k)
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

    @app.get("/stats")
    async def stats() -> dict:
        st: Settings = app.state.settings
        store_stats = await app.state.store.stats()
        return {
            **store_stats,
            "embeddings_backend": st.embeddings_backend,
            "llm_backend": st.llm_backend,
            "embedding_dim": app.state.embedder.dim,
        }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app
