"""Request/response DTOs for the RAG API, plus input bounds and content hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Source = Literal["local", "web", "other"]
# Local ingests default to a high priority so your own data outranks web/other.
DEFAULT_LOCAL_PRIORITY = 100

# Input bounds. These keep a single request's memory/compute footprint finite
# so hostile payloads (a 2 MB document, a 500-document batch, a 50 KB question)
# are rejected with 422 instead of ballooning the in-memory index. The limits
# are ~100x real usage (corpus docs are ~2.5 KB), so they never bite normal use.
MAX_TITLE_CHARS = 1_000
MAX_TEXT_CHARS = 1_000_000  # 1 MB
MAX_QUESTION_CHARS = 10_000
MAX_DOCS_PER_REQUEST = 100
MAX_METADATA_BYTES = 64 * 1024  # 64 KB of JSON-serialised metadata per document


def content_hash(title: str, text: str) -> str:
    """Stable identity of a document's indexed content (title + text)."""
    return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()


class DocumentIn(BaseModel):
    id: str | None = None
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    metadata: dict = Field(default_factory=dict)
    # Local-first provenance. Ingested documents are your own data by default:
    # source="local" + a high priority so they outrank web/other in retrieval.
    source: Source = "local"
    priority: int = Field(default=DEFAULT_LOCAL_PRIORITY, ge=0, le=1000)
    owner: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)

    @field_validator("title", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length=1 lets whitespace-only strings through; reject them so a
        # document that would index to zero chunks never becomes a ghost record.
        if not value.strip():
            raise ValueError("must not be blank or whitespace-only")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_within_bound(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > MAX_METADATA_BYTES:
            raise ValueError(
                f"metadata too large (limit is {MAX_METADATA_BYTES} bytes serialised)"
            )
        return value


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_DOCS_PER_REQUEST)

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> IngestRequest:
        # Two documents sharing an explicit id in one batch is ambiguous: the
        # second would silently overwrite the first's chunks while chunks_indexed
        # still counted both. Reject it so ingest stays deterministic.
        explicit_ids = [d.id for d in self.documents if d.id is not None]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("duplicate document ids within a single request are not allowed")
        return self


class IngestResponse(BaseModel):
    document_ids: list[str]
    chunks_indexed: int
    skipped: int  # documents whose content hash was unchanged (not re-indexed)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=4, ge=1, le=20)
    # Local-first: restrict retrieval to these provenance tiers. None = no filter
    # (all sources, local still boosted). ["local"] enforces "only my data".
    sources: list[Source] | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank or whitespace-only")
        return value


class CitationOut(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    snippet: str
    score: float
    source: str = "local"


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    ord: int
    score: float
    source: str = "local"


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    usage: dict | None = None
