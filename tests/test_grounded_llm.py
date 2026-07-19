"""Grounded mock LLM: grounding + honest abstention, offline and deterministic.

These lock the two production-shaped behaviours ``GroundedMockLLM`` adds over
the plain ``MockLLM``: it only cites retrieved chunks it actually used (so every
``[n]`` resolves), and it abstains when the retrieved context is off-topic
instead of parroting an irrelevant passage. The end-to-end cases drive the real
/query pipeline so citation resolution and the hallucination guard are covered
against a live app, not just the unit.
"""

from __future__ import annotations

import httpx

from app.core.settings import Settings
from app.db.store import ScoredChunk
from app.main import create_app
from app.services.citations import extract_citations
from app.services.llm import (
    NO_CONTEXT_ANSWER,
    NOT_IN_SOURCES_ANSWER,
    GroundedMockLLM,
)

AUTH_HEADERS = {"Authorization": "Bearer demo-key"}


def _chunk(idx: int, content: str, *, doc: str = "doc") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{doc}:{idx}",
        document_id=doc,
        title=doc.title(),
        content=content,
        ord=idx,
        score=1.0 - idx * 0.1,
    )


PGVECTOR_CHUNKS = [
    _chunk(0, "pgvector adds a vector column type to Postgres for similarity search.",
           doc="pg"),
    _chunk(1, "Cosine distance uses the <=> operator with the vector_cosine_ops opclass.",
           doc="pg"),
    _chunk(2, "Espresso extraction depends on grind size, dose and water temperature.",
           doc="coffee"),
]


async def test_grounded_answer_cites_only_used_chunks():
    llm = GroundedMockLLM()
    q = "What operator does pgvector use for cosine distance?"
    result = await llm.answer(q, PGVECTOR_CHUNKS)

    # Cites the two relevant pgvector chunks ([1], [2]); the off-topic espresso
    # chunk ([3]) shares no wording with the question, so it is not cited.
    assert "[1]" in result.answer and "[2]" in result.answer
    assert "[3]" not in result.answer

    # Grounding: every emitted [n] resolves to a real retrieved chunk.
    citations = extract_citations(result.answer, PGVECTOR_CHUNKS)
    assert [c.chunk_id for c in citations] == ["pg:0", "pg:1"]
    assert all(c.document_id == "pg" for c in citations)


async def test_grounded_usage_counts_are_consistent():
    result = await GroundedMockLLM().answer("cosine distance operator", PGVECTOR_CHUNKS)
    usage = result.usage
    assert usage is not None
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    # total is the sum of the parts (a real provider invariant clients rely on)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_honest_abstention_when_context_is_irrelevant():
    # Chunks were retrieved, but none relate to the question -> abstain, no [n].
    result = await GroundedMockLLM().answer(
        "How do I brew a lungo?",
        [_chunk(0, "pgvector adds a vector column type to Postgres.", doc="pg")],
    )
    assert result.answer == NOT_IN_SOURCES_ANSWER
    assert extract_citations(result.answer, PGVECTOR_CHUNKS) == []
    # abstention still reports plausible usage (prompt was built and sent)
    assert result.usage is not None and result.usage["prompt_tokens"] > 0


async def test_empty_retrieval_uses_no_context_answer():
    result = await GroundedMockLLM().answer("anything?", [])
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.usage is not None
    assert result.usage["completion_tokens"] > 0


async def test_grounding_never_emits_out_of_range_index():
    # Many relevant chunks, but max_chunks caps how many are cited; whatever is
    # emitted must still be a valid 1-based index into the retrieved list.
    chunks = [_chunk(i, f"vector search cosine similarity passage {i}", doc="d") for i in range(8)]
    result = await GroundedMockLLM().answer("vector search cosine similarity", chunks)
    citations = extract_citations(result.answer, chunks)
    assert 0 < len(citations) <= GroundedMockLLM.max_chunks
    # No citation dropped as out-of-range: emitted count == resolved count.
    emitted = result.answer.count("[")
    assert emitted == len(citations)


async def test_determinism():
    a = await GroundedMockLLM().answer("cosine distance operator", PGVECTOR_CHUNKS)
    b = await GroundedMockLLM().answer("cosine distance operator", PGVECTOR_CHUNKS)
    assert a.answer == b.answer and a.usage == b.usage


# --------------------------------------------------------------------------- #
# End-to-end through the real /query pipeline (memory store, offline)
# --------------------------------------------------------------------------- #


def _prod_like_client() -> httpx.AsyncClient:
    """App wired with the production-emulating offline stack: semantic embedder
    + grounded LLM over the in-memory store."""
    settings = Settings(
        embeddings_backend="semantic",
        store_backend="memory",
        llm_backend="grounded",
        search_mode="hybrid",
        rag_api_keys="demo-key",
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    )


CORPUS = [
    {"id": "pg", "title": "pgvector", "text":
        "pgvector adds a vector column to Postgres. Cosine distance uses the "
        "<=> operator with the vector_cosine_ops opclass for nearest neighbor search."},
    {"id": "coffee", "title": "Espresso", "text":
        "Espresso extraction depends on grind size, dose and water temperature. "
        "A typical shot uses eighteen grams and a twenty five second pour."},
]


async def test_e2e_grounded_answer_only_cites_retrieved():
    async with _prod_like_client() as client:
        assert (await client.post("/ingest", json={"documents": CORPUS})).status_code == 200
        body = (await client.post(
            "/query",
            json={"question": "Which operator does pgvector use for cosine distance?", "top_k": 4},
        )).json()

    assert body["answer"] != NOT_IN_SOURCES_ANSWER
    assert body["citations"], body
    retrieved_ids = {r["chunk_id"] for r in body["retrieved"]}
    # Grounding at the API boundary: every citation is a chunk that was retrieved.
    for citation in body["citations"]:
        assert citation["chunk_id"] in retrieved_ids
        assert citation["document_id"] == "pg"


async def test_e2e_abstains_on_off_topic_question():
    async with _prod_like_client() as client:
        await client.post("/ingest", json={"documents": CORPUS})
        # A question with no lexical/semantic overlap: retrieval may return rows,
        # but the grounded LLM must refuse rather than cite an unrelated chunk.
        body = (await client.post(
            "/query",
            json={"question": "What is the capital of France?", "top_k": 4},
        )).json()

    assert body["answer"] == NOT_IN_SOURCES_ANSWER
    assert body["citations"] == []


class _HallucinatingLLM:
    """Emits citation indices that point past the retrieved list."""

    async def answer(self, question, chunks):  # noqa: ANN001, ARG002
        from app.services.llm import LLMResult

        return LLMResult(answer="Confident but ungrounded claim [1][7][99].", usage=None)


async def test_e2e_hallucinated_indices_are_dropped():
    settings = Settings(
        embeddings_backend="semantic",
        store_backend="memory",
        llm_backend="mock",
        rag_api_keys="demo-key",
    )
    app = create_app(settings, llm=_HallucinatingLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        await client.post("/ingest", json={"documents": CORPUS})
        body = (await client.post(
            "/query", json={"question": "pgvector cosine operator", "top_k": 2}
        )).json()

    # [7] and [99] are hallucinated (only <=2 chunks retrieved) and dropped;
    # only [1] survives, resolving to a genuinely retrieved chunk.
    n_retrieved = len(body["retrieved"])
    assert n_retrieved <= 2
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == body["retrieved"][0]["chunk_id"]
