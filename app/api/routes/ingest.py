"""Ingest endpoints: JSON documents and file uploads (md/txt/pdf/docx).

Both funnel into ``ingest_documents`` (chunk -> embed -> upsert, idempotent by
content hash). Uploaded files are tagged ``source=local`` by default so your own
data outranks web/other in retrieval.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.deps import require_api_key
from app.schemas import DocumentIn, IngestRequest, IngestResponse, Source
from app.services.extract import (
    MAX_UPLOAD_BYTES,
    ExtractionError,
    UnsupportedFileError,
    extract_text,
)
from app.services.ingest import ingest_documents

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    st = request.app.state.settings
    return await ingest_documents(
        payload.documents,
        store=request.app.state.store,
        embedder=request.app.state.embedder,
        chunk_size=st.chunk_size,
        chunk_overlap=st.chunk_overlap,
    )


@router.post(
    "/ingest/file", response_model=IngestResponse, dependencies=[Depends(require_api_key)]
)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    source: Source = Form(default="local"),
    owner: str | None = Form(default=None),
) -> IngestResponse:
    """Upload a md/txt/pdf/docx file; its text is extracted and ingested."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"file too large (limit {MAX_UPLOAD_BYTES} bytes)"
        )
    try:
        text = extract_text(file.filename or "", data)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not text.strip():
        raise HTTPException(status_code=422, detail="no extractable text found in the file")

    document = DocumentIn(
        title=title or file.filename or "uploaded", text=text, source=source, owner=owner
    )
    st = request.app.state.settings
    return await ingest_documents(
        [document],
        store=request.app.state.store,
        embedder=request.app.state.embedder,
        chunk_size=st.chunk_size,
        chunk_overlap=st.chunk_overlap,
    )
