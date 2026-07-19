import asyncio

import httpx

from app.core.errors import ProviderError
from app.core.settings import Settings
from app.main import create_app
from app.schemas import (
    MAX_DOCS_PER_REQUEST,
    MAX_METADATA_BYTES,
    MAX_QUESTION_CHARS,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
)


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_ingest_validation_422(client):
    # empty document list
    resp = await client.post("/ingest", json={"documents": []})
    assert resp.status_code == 422
    # missing required fields
    resp = await client.post("/ingest", json={"documents": [{"title": "no text"}]})
    assert resp.status_code == 422
    # empty text
    resp = await client.post(
        "/ingest", json={"documents": [{"title": "t", "text": ""}]}
    )
    assert resp.status_code == 422


async def test_query_validation_422(client):
    resp = await client.post("/query", json={})
    assert resp.status_code == 422
    # top_k bounds: 1 <= top_k <= 20
    resp = await client.post("/query", json={"question": "ok", "top_k": 0})
    assert resp.status_code == 422
    resp = await client.post("/query", json={"question": "ok", "top_k": 21})
    assert resp.status_code == 422


async def test_ingest_reports_ids_and_chunks(ingested_client):
    resp = await ingested_client.post(
        "/ingest",
        json={"documents": [{"title": "Auto ID", "text": "Some text without an id."}]},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert len(body["document_ids"]) == 1
    assert body["document_ids"][0]  # generated uuid
    assert body["chunks_indexed"] >= 1


async def test_e2e_query_answers_with_citations(ingested_client):
    resp = await ingested_client.post(
        "/query",
        json={
            "question": "Which operator and opclass does pgvector use for cosine distance?",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"].strip()
    assert "[1]" in body["answer"]

    # retrieval is ranked (non-increasing scores) and the best chunk comes
    # from the right document
    scores = [r["score"] for r in body["retrieved"]]
    assert scores == sorted(scores, reverse=True)
    assert body["retrieved"][0]["document_id"] == "pgvector"

    # citations resolve to the right document
    assert body["citations"], "expected at least one citation"
    top = body["citations"][0]
    assert top["document_id"] == "pgvector"
    assert top["title"] == "pgvector Guide"
    assert top["chunk_id"].startswith("pgvector:")
    assert top["snippet"]
    assert body["usage"]["prompt_tokens"] > 0


async def test_stats_after_ingest(ingested_client):
    resp = await ingested_client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "memory"
    assert body["documents"] == 3
    assert body["chunks"] >= 3
    assert body["embeddings_backend"] == "hash"
    assert body["llm_backend"] == "mock"
    assert body["embedding_dim"] == 256


async def test_query_on_empty_index(client):
    resp = await client.post("/query", json={"question": "anything at all?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved"] == []
    assert body["citations"] == []
    assert body["answer"].startswith("I don't know")


async def test_reingest_replaces_document_chunks(ingested_client):
    resp = await ingested_client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "espresso",
                    "title": "Espresso Handbook v2",
                    "text": (
                        "A lungo pulls a longer shot with twice the water of "
                        "an espresso. Grind coarser to keep extraction balanced."
                    ),
                }
            ]
        },
    )
    assert resp.status_code == 200

    # same id: document count unchanged, old chunks replaced not accumulated
    stats = (await ingested_client.get("/stats")).json()
    assert stats["documents"] == 3
    assert stats["chunks"] == 3

    # the chunk id is reused but must now serve the new content
    resp = await ingested_client.post(
        "/query", json={"question": "How much water does a lungo use?", "top_k": 1}
    )
    top = resp.json()["citations"][0]
    assert top["chunk_id"] == "espresso:0"
    assert top["title"] == "Espresso Handbook v2"
    assert "lungo" in top["snippet"]
    assert "Crema" not in top["snippet"]


# --------------------------------------------------------------------------- #
# Hostile-input hardening (adversarial regression tests)
# --------------------------------------------------------------------------- #


async def test_ingest_oversized_inputs_rejected_422(client):
    # text over the 1 MB cap (a 2 MB document would otherwise index ~3k chunks)
    resp = await client.post(
        "/ingest", json={"documents": [{"title": "big", "text": "x" * (MAX_TEXT_CHARS + 1)}]}
    )
    assert resp.status_code == 422
    # title over cap
    resp = await client.post(
        "/ingest",
        json={"documents": [{"title": "T" * (MAX_TITLE_CHARS + 1), "text": "ok"}]},
    )
    assert resp.status_code == 422
    # too many documents in one batch
    docs = [{"title": f"d{i}", "text": "hi"} for i in range(MAX_DOCS_PER_REQUEST + 1)]
    resp = await client.post("/ingest", json={"documents": docs})
    assert resp.status_code == 422
    # metadata over the serialised-size cap
    resp = await client.post(
        "/ingest",
        json={
            "documents": [
                {"title": "m", "text": "hi", "metadata": {"a": "x" * (MAX_METADATA_BYTES + 10)}}
            ]
        },
    )
    assert resp.status_code == 422


async def test_ingest_blank_text_or_title_rejected_422(client):
    # whitespace-only text passes min_length=1 but must not create a ghost doc
    resp = await client.post(
        "/ingest", json={"documents": [{"title": "t", "text": "   \n\t  "}]}
    )
    assert resp.status_code == 422
    resp = await client.post(
        "/ingest", json={"documents": [{"title": "   ", "text": "real text"}]}
    )
    assert resp.status_code == 422


async def test_ingest_duplicate_ids_in_one_request_rejected_422(client):
    resp = await client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "dup", "title": "A", "text": "first alpha"},
                {"id": "dup", "title": "B", "text": "second beta"},
            ]
        },
    )
    assert resp.status_code == 422
    # nothing partially ingested
    stats = (await client.get("/stats")).json()
    assert stats["documents"] == 0 and stats["chunks"] == 0


async def test_ingest_separator_only_text_accepted(client):
    # non-blank (contains '.') so it is a legitimate single chunk, not a ghost doc
    resp = await client.post(
        "/ingest", json={"documents": [{"title": "seps", "text": "\n\n. \n\n"}]}
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] == 1


async def test_ingest_metadata_within_bound_accepted(client):
    resp = await client.post(
        "/ingest",
        json={"documents": [{"title": "m", "text": "hi", "metadata": {"tags": "x" * 1000}}]},
    )
    assert resp.status_code == 200


async def test_query_oversized_question_rejected_422(client):
    resp = await client.post("/query", json={"question": "q" * (MAX_QUESTION_CHARS + 1)})
    assert resp.status_code == 422
    # blank question also rejected
    resp = await client.post("/query", json={"question": "   "})
    assert resp.status_code == 422


async def test_query_unicode_and_injection_are_inert(ingested_client):
    # unicode / emoji / RTL question does not crash retrieval
    resp = await ingested_client.post(
        "/query", json={"question": "café ☕ مرحبا שלום 你好 🚀 pgvector"}
    )
    assert resp.status_code == 200
    # SQL-injection-looking text is just tokens to the memory store: inert
    before = (await ingested_client.get("/stats")).json()
    resp = await ingested_client.post(
        "/query", json={"question": "'; DROP TABLE chunks; -- SELECT * FROM x"}
    )
    assert resp.status_code == 200
    after = (await ingested_client.get("/stats")).json()
    assert before["documents"] == after["documents"]
    assert before["chunks"] == after["chunks"]


async def test_concurrent_ingest_and_query_keep_store_coherent(client):
    async def ingest(i: int):
        return await client.post(
            "/ingest",
            json={
                "documents": [
                    {"id": f"doc{i}", "title": f"T{i}", "text": f"alpha{i} beta{i} gamma{i}. " * 20}
                ]
            },
        )

    async def query(i: int):
        return await client.post("/query", json={"question": f"alpha{i} beta{i}", "top_k": 5})

    tasks = []
    for i in range(20):
        tasks.append(ingest(i))
        tasks.append(query(i))
    results = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in results)

    reported = sum(r.json()["chunks_indexed"] for r in results if "chunks_indexed" in r.json())
    stats = (await client.get("/stats")).json()
    assert stats["documents"] == 20
    # store chunk count equals the sum of per-document chunks reported at ingest
    assert stats["chunks"] == reported


class _FailingLLM:
    async def answer(self, question, chunks):
        raise ProviderError("gateway timed out")


async def test_llm_failure_maps_to_502(store, embedder):
    app = create_app(
        Settings(embeddings_backend="hash", store_backend="memory", llm_backend="mock"),
        store=store,
        embedder=embedder,
        llm=_FailingLLM(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": "Bearer demo-key"}
    ) as client:
        resp = await client.post("/query", json={"question": "boom"})
    assert resp.status_code == 502
    assert "upstream provider error" in resp.json()["detail"]
