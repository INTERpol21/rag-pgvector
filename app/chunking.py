"""Recursive character text splitting.

A pure, dependency-free splitter in the spirit of LangChain's
``RecursiveCharacterTextSplitter``: windows of at most ``chunk_size``
characters, cut at the highest-priority separator available inside the
window, with a fixed ``chunk_overlap`` carried into the next chunk.

Guarantees:

* every chunk is an exact contiguous substring of the input;
* consecutive chunks overlap by exactly ``chunk_overlap`` characters
  (the final chunk simply extends to the end of the text);
* no content is lost: ``chunks[0] + "".join(c[overlap:] for c in chunks[1:])``
  reconstructs the original text (whitespace-only chunks excepted — they
  are dropped).
"""

from __future__ import annotations

from typing import Sequence

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def _split_point(
    text: str,
    start: int,
    end: int,
    separators: Sequence[str],
    overlap: int,
) -> int:
    """Pick a cut position in ``(start, end]``.

    Tries separators in priority order and takes the *rightmost* occurrence
    that still leaves the chunk longer than ``overlap`` (so the sliding
    window always makes forward progress). Falls back to a hard cut at
    ``end`` when no separator qualifies.
    """
    for sep in separators:
        idx = text.rfind(sep, start, end)
        if idx == -1:
            continue
        candidate = idx + len(sep)
        if candidate - start > overlap:
            return candidate
    return end


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` chars.

    Edge cases: empty / whitespace-only input yields ``[]``; text of length
    ``<= chunk_size`` (including the exact-boundary case) yields a single
    chunk equal to the input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    if not text or not text.strip():
        return []

    n = len(text)
    if n <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        split = n if end == n else _split_point(text, start, end, separators, chunk_overlap)
        piece = text[start:split]
        if piece.strip():
            chunks.append(piece)
        if split >= n:
            break
        start = split - chunk_overlap  # split - start > overlap => strict progress
    return chunks
