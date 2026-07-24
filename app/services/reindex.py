"""Detect an embedder switch at startup and re-embed the stored corpus.

Ingest dedups by content hash, so after switching ``EMBEDDINGS_BACKEND`` or
``EMBEDDING_MODEL`` a re-ingest of unchanged documents is skipped — the old
model's vectors stay in place. When the width also changes the schema guard
catches it, but a same-dim switch silently mixes incompatible vector spaces
and retrieval quality falls off a cliff with no error anywhere.

The store therefore records the :attr:`Embedder.fingerprint` that wrote it.
On startup :func:`sync_embedder_fingerprint` compares it with the active
embedder and, on mismatch, re-embeds every chunk in place. Runs before the
app serves traffic; a failing embedder fails startup (better than serving
mixed vectors).
"""

from __future__ import annotations

from app.db.store import VectorStore
from app.services.embeddings import Embedder

# Bounds the request size for remote embedders (gateway/openai); the offline
# embedders don't care.
EMBED_BATCH_SIZE = 64


async def sync_embedder_fingerprint(store: VectorStore, embedder: Embedder) -> str:
    """Reconcile the store's vectors with the active embedder; return the outcome.

    Outcomes: ``match`` (nothing to do), ``adopted`` (store had no recorded
    fingerprint — fresh, or written before fingerprints existed; existing
    vectors are assumed to match the running config, which is exactly the
    pre-fingerprint status quo), or ``reembedded:<n>`` (fingerprint mismatch;
    all ``n`` chunks were re-embedded with the active embedder).
    """
    stored = await store.embedder_fingerprint()
    if stored == embedder.fingerprint:
        return "match"
    if stored is None:
        await store.set_embedder_fingerprint(embedder.fingerprint)
        return "adopted"

    # N replicas booting after a switch would otherwise ALL re-embed the
    # corpus (N x the gateway bill) and interleave their writes. The store's
    # reindex lock (pgvector: pg_advisory_lock) admits one worker; the rest
    # block here, then the re-check finds the work already done.
    async with store.reindex_lock():
        stored = await store.embedder_fingerprint()
        if stored == embedder.fingerprint:
            return "match:after-wait"

        chunk_texts = await store.all_chunk_texts()
        for start in range(0, len(chunk_texts), EMBED_BATCH_SIZE):
            batch = chunk_texts[start : start + EMBED_BATCH_SIZE]
            vectors = await embedder.embed([content for _, content in batch])
            await store.update_embeddings(
                {chunk_id: vector for (chunk_id, _), vector in zip(batch, vectors, strict=True)}
            )
        await store.set_embedder_fingerprint(embedder.fingerprint)
        return f"reembedded:{len(chunk_texts)}"
