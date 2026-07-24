"""Embedder-switch detection: fingerprints and the startup re-embed.

The trap being closed: content-hash dedup skips unchanged documents on
ingest, so switching EMBEDDINGS_BACKEND/EMBEDDING_MODEL (same dim) used to
leave the old model's vectors in place and silently mix vector spaces.
"""

from __future__ import annotations

from app.core.settings import Settings
from app.db.store import ChunkRecord, DocumentRecord, MemoryVectorStore
from app.services.embeddings import (
    HashingEmbedder,
    SemanticMockEmbedder,
    build_embedder,
)
from app.services.reindex import EMBED_BATCH_SIZE, sync_embedder_fingerprint


def test_fingerprints_identify_the_vector_space():
    assert HashingEmbedder(dim=8).fingerprint == "hash:8"
    assert SemanticMockEmbedder(dim=8).fingerprint == "semantic:8"
    assert HashingEmbedder(dim=8).fingerprint != HashingEmbedder(dim=16).fingerprint


def test_gateway_and_openai_backends_share_a_fingerprint():
    """Same model through the gateway or direct produces the same vectors —
    switching the transport must not trigger a corpus re-embed."""
    base = {"embedding_model": "text-embedding-3-small", "embedding_dim": 256}
    via_gateway = build_embedder(Settings(embeddings_backend="gateway", **base))
    direct = build_embedder(Settings(embeddings_backend="openai", **base))
    assert via_gateway.fingerprint == direct.fingerprint == "openai:text-embedding-3-small:256"


async def _ingest(store: MemoryVectorStore, embedder: HashingEmbedder, texts: list[str]) -> None:
    vectors = await embedder.embed(texts)
    for i, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
        doc = DocumentRecord(id=f"d{i}", title=f"doc {i}", content_hash=f"h{i}")
        await store.upsert(doc, [ChunkRecord(f"d{i}:0", f"d{i}", 0, text, vector)])


async def test_fresh_store_adopts_the_fingerprint():
    store = MemoryVectorStore()
    embedder = HashingEmbedder(dim=8)
    assert await store.embedder_fingerprint() is None
    assert await sync_embedder_fingerprint(store, embedder) == "adopted"
    assert await store.embedder_fingerprint() == "hash:8"


async def test_matching_fingerprint_is_a_noop():
    store = MemoryVectorStore()
    embedder = HashingEmbedder(dim=8)
    await sync_embedder_fingerprint(store, embedder)
    assert await sync_embedder_fingerprint(store, embedder) == "match"


async def test_pre_fingerprint_corpus_is_adopted_without_reembedding():
    """Upgrade path: vectors written before fingerprints existed are assumed
    to match the running config (the status quo), not rewritten."""
    store = MemoryVectorStore()
    old = HashingEmbedder(dim=8)
    await _ingest(store, old, ["alpha beta"])
    original = store._chunks["d0:0"].embedding

    assert await sync_embedder_fingerprint(store, old) == "adopted"
    assert store._chunks["d0:0"].embedding == original


async def test_switched_embedder_reembeds_every_chunk():
    store = MemoryVectorStore()
    old = HashingEmbedder(dim=8)
    await _ingest(store, old, ["alpha beta", "gamma delta"])
    await sync_embedder_fingerprint(store, old)

    new = SemanticMockEmbedder(dim=8)  # same dim: the schema guard cannot see this
    outcome = await sync_embedder_fingerprint(store, new)

    assert outcome == "reembedded:2"
    assert await store.embedder_fingerprint() == "semantic:8"
    expected = await new.embed(["alpha beta", "gamma delta"])
    assert store._chunks["d0:0"].embedding == expected[0]
    assert store._chunks["d1:0"].embedding == expected[1]


async def test_reembed_handles_more_chunks_than_one_batch():
    store = MemoryVectorStore()
    old = HashingEmbedder(dim=8)
    texts = [f"unique token{i}" for i in range(EMBED_BATCH_SIZE + 3)]
    await _ingest(store, old, texts)
    await sync_embedder_fingerprint(store, old)

    outcome = await sync_embedder_fingerprint(store, SemanticMockEmbedder(dim=8))
    assert outcome == f"reembedded:{EMBED_BATCH_SIZE + 3}"


async def test_search_works_in_the_new_space_after_reembed():
    """The point of the exercise: post-switch queries embedded by the new
    model must match the re-embedded corpus."""
    store = MemoryVectorStore()
    await _ingest(store, HashingEmbedder(dim=64), ["the quick brown fox", "vector databases"])
    await sync_embedder_fingerprint(store, HashingEmbedder(dim=64))

    new = SemanticMockEmbedder(dim=64)
    await sync_embedder_fingerprint(store, new)

    query = (await new.embed(["quick brown fox"]))[0]
    results = await store.search(query, top_k=1)
    assert results[0].chunk_id == "d0:0"
    assert results[0].score > 0.5


async def test_second_replica_finds_the_work_done_under_the_lock():
    """Cross-replica dedup: if another worker finished the re-embed while we
    waited on the reindex lock, the re-check returns without re-embedding."""

    class RacedStore(MemoryVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.winner_fingerprint: str | None = None

        def reindex_lock(self):  # type: ignore[override]
            outer = super().reindex_lock()

            class _Lock:
                async def __aenter__(inner) -> None:
                    await outer.__aenter__()
                    # Simulate the other replica finishing while we waited.
                    if self.winner_fingerprint is not None:
                        self._embedder_fingerprint = self.winner_fingerprint

                async def __aexit__(inner, *exc) -> bool | None:
                    return await outer.__aexit__(*exc)

            return _Lock()

    store = RacedStore()
    old = HashingEmbedder(dim=8)
    await _ingest(store, old, ["alpha beta"])
    await sync_embedder_fingerprint(store, old)
    original = store._chunks["d0:0"].embedding

    new = SemanticMockEmbedder(dim=8)
    store.winner_fingerprint = new.fingerprint  # "другая реплика" уже всё сделала
    assert await sync_embedder_fingerprint(store, new) == "match:after-wait"
    assert store._chunks["d0:0"].embedding == original  # мы НЕ переэмбеддили
