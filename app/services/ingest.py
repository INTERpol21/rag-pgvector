"""Core ingest pipeline: chunk -> embed -> upsert, idempotent by content hash.

Shared by the JSON ``/ingest`` endpoint and the ``/ingest/file`` upload endpoint
so both go through exactly the same chunking, batched embedding, content-hash
dedup and upsert path.
"""

from __future__ import annotations

import uuid

from app.db.store import ChunkRecord, DocumentRecord, VectorStore
from app.schemas import DocumentIn, IngestResponse, content_hash
from app.services.chunking import chunk_text
from app.services.embeddings import Embedder


async def ingest_documents(
    documents: list[DocumentIn],
    *,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResponse:
    """Chunk, embed and upsert ``documents``; skip any whose content hash is unchanged."""
    document_ids: list[str] = []
    incoming: list[tuple[DocumentIn, str, str]] = []  # (doc, id, hash)
    for doc in documents:
        doc_id = doc.id or uuid.uuid4().hex
        document_ids.append(doc_id)
        incoming.append((doc, doc_id, content_hash(doc.title, doc.text)))

    # Idempotence: a document whose (title, text) hash matches what is already
    # stored is skipped before any chunking or embedding work.
    existing = await store.content_hashes([i[1] for i in incoming])
    skipped = 0
    docs_with_chunks: list[tuple[DocumentRecord, list[str]]] = []
    all_texts: list[str] = []
    for doc, doc_id, doc_hash in incoming:
        if existing.get(doc_id) == doc_hash:
            skipped += 1
            continue
        pieces = chunk_text(doc.text, chunk_size, chunk_overlap)
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
    vectors = await embedder.embed(all_texts) if all_texts else []

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
        await store.upsert(record, chunk_records)
        chunks_indexed += len(chunk_records)

    return IngestResponse(
        document_ids=document_ids, chunks_indexed=chunks_indexed, skipped=skipped
    )
