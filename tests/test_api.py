import httpx

from app.errors import ProviderError
from app.main import create_app
from app.settings import Settings


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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", json={"question": "boom"})
    assert resp.status_code == 502
    assert "upstream provider error" in resp.json()["detail"]
