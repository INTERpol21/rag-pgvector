from app.store import ChunkRecord, DocumentRecord, MemoryVectorStore


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
