"""Liveness probe."""

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")
