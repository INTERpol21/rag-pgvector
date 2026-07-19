"""Query endpoint: embed -> retrieve (+rerank) -> synthesize with [n] citations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_api_key
from app.db.store import search_with_mode
from app.schemas import CitationOut, QueryRequest, QueryResponse, RetrievedChunkOut
from app.services.citations import extract_citations

router = APIRouter()


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    state = request.app.state
    st = state.settings
    query_vec = (await state.embedder.embed([payload.question]))[0]
    retrieved = await search_with_mode(
        state.store,
        st.search_mode,
        query_vec,
        payload.question,
        payload.top_k,
        sources=set(payload.sources) if payload.sources else None,
    )
    if state.reranker is not None:
        retrieved = await state.reranker.rerank(payload.question, retrieved)
    result = await state.llm.answer(payload.question, retrieved)
    citations = extract_citations(result.answer, retrieved)
    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in citations],
        retrieved=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                ord=c.ord,
                score=c.score,
                source=c.source,
            )
            for c in retrieved
        ],
        usage=result.usage,
    )
