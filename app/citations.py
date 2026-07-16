"""Map inline [n] references in an LLM answer back to retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.store import ScoredChunk

_REF_RE = re.compile(r"\[(\d+)\]")

SNIPPET_LEN = 200


@dataclass(frozen=True)
class Citation:
    document_id: str
    title: str
    chunk_id: str
    snippet: str
    score: float


def _snippet(content: str, limit: int = SNIPPET_LEN) -> str:
    flat = " ".join(content.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def extract_reference_indices(answer: str) -> list[int]:
    """Return 1-based [n] indices in order of first appearance, deduplicated."""
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _REF_RE.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def extract_citations(
    answer: str, retrieved: Sequence[ScoredChunk]
) -> list[Citation]:
    """Resolve [n] references against the retrieved chunks (1-based).

    References that point outside the retrieved list (hallucinated indices,
    [0], etc.) are silently dropped; duplicates collapse to one citation,
    preserving first-appearance order.
    """
    citations: list[Citation] = []
    for n in extract_reference_indices(answer):
        if 1 <= n <= len(retrieved):
            chunk = retrieved[n - 1]
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    chunk_id=chunk.chunk_id,
                    snippet=_snippet(chunk.content),
                    score=chunk.score,
                )
            )
    return citations
