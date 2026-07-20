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
    resp = await client.post("/v1/ingest", json={"documents": [WEB_DOC, LOCAL_DOC]})
    assert resp.status_code == 200, resp.text

    body = (await client.post("/v1/query", json={"question": QUESTION, "top_k": 4})).json()
    assert body["retrieved"], body
    # Higher-priority local document wins the top slot regardless of raw score.
    assert body["retrieved"][0]["source"] == "local"


async def test_strict_local_excludes_web(client):
    await client.post("/v1/ingest", json={"documents": [WEB_DOC, LOCAL_DOC]})

    body = (
        await client.post(
            "/v1/query", json={"question": QUESTION, "top_k": 4, "sources": ["local"]}
        )
    ).json()
    assert body["retrieved"], body
    assert all(chunk["source"] == "local" for chunk in body["retrieved"])
    assert all(chunk["source"] != "web" for chunk in body["retrieved"])


async def test_ingest_defaults_to_local_source(client):
    resp = await client.post(
        "/v1/ingest",
        json={"documents": [{"id": "d", "title": "T", "text": QUESTION}]},
    )
    assert resp.status_code == 200

    body = (await client.post("/v1/query", json={"question": QUESTION, "top_k": 4})).json()
    assert body["retrieved"][0]["source"] == "local"


async def test_citations_carry_source(client):
    await client.post("/v1/ingest", json={"documents": [LOCAL_DOC]})
    body = (await client.post("/v1/query", json={"question": QUESTION, "top_k": 4})).json()
    # The extractive mock LLM cites; every citation must expose its provenance.
    for citation in body["citations"]:
        assert citation["source"] in ("local", "web", "other")


def test_priority_defaults_follow_source():
    """Omitted priority is derived from source, not always the local default.

    Regression: a web document ingested without an explicit priority used to
    inherit priority=100 and tie with local data, silently breaking local-first.
    """
    from app.schemas import DocumentIn

    assert DocumentIn(title="t", text="x", source="local").priority == 100
    assert DocumentIn(title="t", text="x", source="other").priority == 50
    assert DocumentIn(title="t", text="x", source="web").priority == 0
    # An explicit priority always wins over the source-derived default.
    assert DocumentIn(title="t", text="x", source="web", priority=99).priority == 99
    # Omitted source defaults to local -> high priority.
    assert DocumentIn(title="t", text="x").priority == 100


async def test_local_outranks_web_without_explicit_priority(client):
    """End-to-end: same text, only source differs, no priority set -> local wins."""
    text = "pgvector cosine distance uses the <=> operator with vector_cosine_ops."
    await client.post(
        "/v1/ingest",
        json={
            "documents": [
                {"id": "w", "title": "Web", "text": text, "source": "web"},
                {"id": "l", "title": "Local", "text": text, "source": "local"},
            ]
        },
    )
    body = (await client.post("/v1/query", json={"question": text, "top_k": 4})).json()
    assert body["retrieved"][0]["source"] == "local"
