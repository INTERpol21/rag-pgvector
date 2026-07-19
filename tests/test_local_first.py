"""Local-first retrieval: your data outranks web, and strict mode excludes web."""

from __future__ import annotations

# Identical content, different provenance + priority: isolates the local-first
# ranking from raw similarity (both chunks are equally relevant to the query).
LOCAL_DOC = {
    "id": "loc",
    "title": "Local pgvector notes",
    "text": "pgvector cosine distance uses the <=> operator with vector_cosine_ops.",
    "source": "local",
    "priority": 100,
}
WEB_DOC = {
    "id": "web",
    "title": "Web pgvector page",
    "text": "pgvector cosine distance uses the <=> operator with vector_cosine_ops.",
    "source": "web",
    "priority": 0,
}
QUESTION = "pgvector cosine distance operator vector_cosine_ops"


async def test_local_is_boosted_above_web(client):
    resp = await client.post("/ingest", json={"documents": [WEB_DOC, LOCAL_DOC]})
    assert resp.status_code == 200, resp.text

    body = (await client.post("/query", json={"question": QUESTION, "top_k": 4})).json()
    assert body["retrieved"], body
    # Higher-priority local document wins the top slot regardless of raw score.
    assert body["retrieved"][0]["source"] == "local"


async def test_strict_local_excludes_web(client):
    await client.post("/ingest", json={"documents": [WEB_DOC, LOCAL_DOC]})

    body = (
        await client.post(
            "/query", json={"question": QUESTION, "top_k": 4, "sources": ["local"]}
        )
    ).json()
    assert body["retrieved"], body
    assert all(chunk["source"] == "local" for chunk in body["retrieved"])
    assert all(chunk["source"] != "web" for chunk in body["retrieved"])


async def test_ingest_defaults_to_local_source(client):
    resp = await client.post(
        "/ingest",
        json={"documents": [{"id": "d", "title": "T", "text": QUESTION}]},
    )
    assert resp.status_code == 200

    body = (await client.post("/query", json={"question": QUESTION, "top_k": 4})).json()
    assert body["retrieved"][0]["source"] == "local"


async def test_citations_carry_source(client):
    await client.post("/ingest", json={"documents": [LOCAL_DOC]})
    body = (await client.post("/query", json={"question": QUESTION, "top_k": 4})).json()
    # The extractive mock LLM cites; every citation must expose its provenance.
    for citation in body["citations"]:
        assert citation["source"] in ("local", "web", "other")
