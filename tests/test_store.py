from app.db.store import ChunkRecord, DocumentRecord, MemoryVectorStore


def _chunk(cid: str, doc: str, ord_: int, content: str, emb: list[float]) -> ChunkRecord:
    return ChunkRecord(id=cid, document_id=doc, ord=ord_, content=content, embedding=emb)


async def test_memory_store_ranks_by_cosine():
    store = MemoryVectorStore()
    await store.upsert(
        DocumentRecord(id="d1", title="Doc One"),
        [
            _chunk("d1:0", "d1", 0, "exact match", [1.0, 0.0, 0.0]),
            _chunk("d1:1", "d1", 1, "close match", [0.9, 0.1, 0.0]),
        ],
    )
    await store.upsert(
        DocumentRecord(id="d2", title="Doc Two"),
        [_chunk("d2:0", "d2", 0, "orthogonal", [0.0, 1.0, 0.0])],
    )

    results = await store.search([1.0, 0.0, 0.0], top_k=3)
    assert [r.chunk_id for r in results] == ["d1:0", "d1:1", "d2:0"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].title == "Doc One"

    # top_k is respected
    assert len(await store.search([1.0, 0.0, 0.0], top_k=2)) == 2


async def test_memory_store_upsert_replaces_and_stats():
    store = MemoryVectorStore()
    doc = DocumentRecord(id="d1", title="Doc")
    await store.upsert(
        doc,
        [
            _chunk("d1:0", "d1", 0, "a", [1.0, 0.0]),
            _chunk("d1:1", "d1", 1, "b", [0.0, 1.0]),
        ],
    )
    # re-ingest with a single chunk: old chunks must not linger (cascade semantics)
    await store.upsert(doc, [_chunk("d1:0", "d1", 0, "a2", [1.0, 0.0])])

    stats = await store.stats()
    assert stats == {"backend": "memory", "documents": 1, "chunks": 1}
    results = await store.search([0.0, 1.0], top_k=10)
    assert [r.chunk_id for r in results] == ["d1:0"]
    assert results[0].content == "a2"


async def test_bm25_handles_non_latin_corpus():
    """An all-Cyrillic corpus used to crash BM25 with ZeroDivisionError.

    The old ASCII-only tokenizer gave every chunk zero tokens, so avg_len was 0
    and the length normalisation divided by it — every /v1/query against a
    Russian corpus was a 500 in the default hybrid mode. The tokenizer is now
    Unicode-aware, so the same corpus must be searchable by keyword.
    """
    store = MemoryVectorStore()
    await store.upsert(
        DocumentRecord(id="ru", title="Заметка"),
        [_chunk("ru:0", "ru", 0, "Привет мир это заметка о векторном поиске", [1.0, 0.0])],
    )

    # exact word forms: BM25 has no stemming, only Unicode tokenization
    results = await store.search_bm25("заметка мир", top_k=4)
    assert [r.chunk_id for r in results] == ["ru:0"]

    # Mixed-script queries must not crash even when only one leg matches.
    assert await store.search_bm25("привет hello", top_k=4)


async def test_bm25_empty_token_corpus_returns_no_matches():
    """Punctuation-only chunks tokenize to nothing; BM25 must return [] not 500."""
    store = MemoryVectorStore()
    await store.upsert(
        DocumentRecord(id="p", title="Punct"),
        [_chunk("p:0", "p", 0, "!!! ??? --- ...", [1.0, 0.0])],
    )
    assert await store.search_bm25("anything", top_k=4) == []
