"""Stats endpoint: store counts plus the active backends."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_api_key

router = APIRouter()


@router.get("/stats", dependencies=[Depends(require_api_key)])
async def stats(request: Request) -> dict:
    state = request.app.state
    st = state.settings
    store_stats = await state.store.stats()
    return {
        **store_stats,
        "embeddings_backend": st.embeddings_backend,
        "llm_backend": st.llm_backend,
        "embedding_dim": state.embedder.dim,
        "search_mode": st.search_mode,
        "reranker": state.reranker.name if state.reranker else "none",
    }
