"""Ingest endpoints: JSON documents and file uploads (md/txt/pdf/docx).

Both funnel into ``ingest_documents`` (chunk -> embed -> upsert, idempotent by
content hash). Uploaded files are tagged ``source=local`` by default so your own
data outranks web/other in retrieval.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.api.deps import require_api_key
from app.schemas import MAX_TEXT_CHARS, DocumentIn, IngestRequest, IngestResponse, Source
from app.services.extract import (
    MAX_UPLOAD_BYTES,
    ExtractionError,
    UnsupportedFileError,
    extract_text,
)
from app.services.ingest import ingest_documents

router = APIRouter()

# Read the upload in bounded chunks so an oversized/malicious body is rejected
# before it is fully materialised in memory (a naive ``await file.read()`` buffers
# the entire request body first, giving a trivial memory-exhaustion DoS).
_UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _read_capped(file: UploadFile, limit: int) -> bytes | None:
    """Read up to ``limit`` bytes; return ``None`` if the file exceeds it."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


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
    data = await _read_capped(file, MAX_UPLOAD_BYTES)
    if data is None:
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
    if len(text) > MAX_TEXT_CHARS:
        # A large file whose extracted text exceeds the per-document limit must
        # return a clean 413, not crash the DocumentIn construction into a 500.
        raise HTTPException(
            status_code=413,
            detail=f"extracted text too large ({len(text)} chars; limit {MAX_TEXT_CHARS})",
        )

    try:
        document = DocumentIn(
            title=title or file.filename or "uploaded", text=text, source=source, owner=owner
        )
    except ValidationError as exc:
        # Any other schema violation (e.g. an over-long filename as title) becomes
        # a 422 with the field errors, never an unhandled 500. Report loc+msg only
        # (not the input value, which could be large).
        errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        raise HTTPException(status_code=422, detail=errors) from exc
    st = request.app.state.settings
    return await ingest_documents(
        [document],
        store=request.app.state.store,
        embedder=request.app.state.embedder,
        chunk_size=st.chunk_size,
        chunk_overlap=st.chunk_overlap,
    )
