"""Ingest endpoint: chunk -> embed -> upsert, idempotent by content hash."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_api_key
from app.db.store import ChunkRecord, DocumentRecord
from app.schemas import DocumentIn, IngestRequest, IngestResponse, content_hash
from app.services.chunking import chunk_text

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    state = request.app.state
    st = state.settings
    document_ids: list[str] = []
    incoming: list[tuple[DocumentIn, str, str]] = []  # (doc, id, hash)

    for doc in payload.documents:
        doc_id = doc.id or uuid.uuid4().hex
        document_ids.append(doc_id)
        incoming.append((doc, doc_id, content_hash(doc.title, doc.text)))

    # Idempotence: a document whose (title, text) hash matches what is
    # already stored is skipped before any chunking or embedding work.
    existing = await state.store.content_hashes([i[1] for i in incoming])
    skipped = 0
    docs_with_chunks: list[tuple[DocumentRecord, list[str]]] = []
    all_texts: list[str] = []
    for doc, doc_id, doc_hash in incoming:
        if existing.get(doc_id) == doc_hash:
            skipped += 1
            continue
        pieces = chunk_text(doc.text, st.chunk_size, st.chunk_overlap)
        record = DocumentRecord(
            id=doc_id,
            title=doc.title,
            metadata=doc.metadata,
            content_hash=doc_hash,
            source=doc.source,
            priority=doc.priority,
            owner=doc.owner,
        )
        docs_with_chunks.append((record, pieces))
        all_texts.extend(pieces)

    # One batched embedding call for the whole request.
    vectors = await state.embedder.embed(all_texts) if all_texts else []

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
        await state.store.upsert(record, chunk_records)
        chunks_indexed += len(chunk_records)

    return IngestResponse(
        document_ids=document_ids, chunks_indexed=chunks_indexed, skipped=skipped
    )
