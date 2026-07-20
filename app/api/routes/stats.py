"""Stats endpoint: store counts plus the active backends."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_api_key
from app.schemas import StatsResponse

router = APIRouter()


@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
async def stats(request: Request) -> StatsResponse:
    state = request.app.state
    st = state.settings
    store_stats = await state.store.stats()
    return StatsResponse(
        backend=store_stats["backend"],
        documents=store_stats["documents"],
        chunks=store_stats["chunks"],
        embeddings_backend=st.embeddings_backend,
        llm_backend=st.llm_backend,
        embedding_dim=state.embedder.dim,
        search_mode=st.search_mode,
        reranker=state.reranker.name if state.reranker else "none",
    )
