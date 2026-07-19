"""Optional second-stage reranking of retrieved chunks.

Runs between retrieval and synthesis: takes the top-k ``ScoredChunk`` list
and reorders it by estimated relevance to the question. Chunks keep their
retrieval scores — a reranker changes only the order, so score provenance
stays honest.

``RERANKER`` selects the stage: ``none`` (default, stage skipped), ``mock``
(offline lexical overlap) or ``llm`` (one batched scoring call through the
OpenAI-compatible chat client).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.llm import LLMResult, OpenAIChatLLM
from app.store import ScoredChunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]: ...


class MockReranker:
    """Deterministic lexical-overlap reranker for offline demos and tests.

    Orders chunks by the fraction of the question's content words they
    contain (original rank breaks ties). This is not a cross-encoder: no
    semantics, no learned relevance — it exists so the rerank stage can be
    exercised without a model or network, with fully reproducible output.
    """

    name = "mock"

    async def rerank(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]:
        question_words = _content_words(question)
        if not question_words:
            return list(chunks)

        def overlap(chunk: ScoredChunk) -> float:
            return len(question_words & _content_words(chunk.content)) / len(question_words)

        indexed = list(enumerate(chunks))
        indexed.sort(key=lambda pair: (-overlap(pair[1]), pair[0]))
        return [chunk for _, chunk in indexed]


RERANK_SYSTEM_PROMPT = (
    "You are a search relevance judge. For each numbered passage, rate how "
    "well it answers the question on a 0-10 scale (10 = directly answers, "
    "0 = unrelated). Reply with one line per passage in the exact form "
    "`<number>: <score>` and nothing else."
)

# Accepts "1: 7", "[2] - 3.5", "3. 10" etc.; ignores everything else.
_SCORE_LINE_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.\-]\s*(\d+(?:\.\d+)?)\s*$")


def parse_scores(reply: str, n_candidates: int) -> dict[int, float] | None:
    """Parse ``<number>: <score>`` lines into {0-based index: score}.

    Out-of-range indices and out-of-range scores are dropped; duplicates keep
    the first occurrence. Returns ``None`` when nothing usable was parsed, so
    the caller can fall back to the original order.
    """
    scores: dict[int, float] = {}
    for line in reply.splitlines():
        match = _SCORE_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1  # prompt numbers passages from 1
        score = float(match.group(2))
        if 0 <= index < n_candidates and 0.0 <= score <= 10.0 and index not in scores:
            scores[index] = score
    return scores or None


class LLMReranker:
    """Rerank with one batched 0-10 scoring prompt through the chat client.

    Sends all candidates in a single request, parses per-passage scores and
    reorders by them (unscored candidates sink below scored ones; original
    rank breaks ties). An unparsable reply falls back to the original order;
    transport failures raise ``ProviderError`` like every other LLM call.
    """

    name = "llm"

    def __init__(self, llm: OpenAIChatLLM) -> None:
        self._llm = llm

    @staticmethod
    def _messages(question: str, chunks: Sequence[ScoredChunk]) -> list[dict]:
        passages = "\n\n".join(
            f"[{i}] {chunk.content.strip()}" for i, chunk in enumerate(chunks, start=1)
        )
        user = f"Question: {question}\n\nPassages:\n{passages}\n\nScores:"
        return [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    async def rerank(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        result: LLMResult = await self._llm.complete(self._messages(question, chunks))
        scores = parse_scores(result.answer, len(chunks))
        if scores is None:
            return list(chunks)
        indexed = list(enumerate(chunks))
        indexed.sort(key=lambda pair: (-scores.get(pair[0], -1.0), pair[0]))
        return [chunk for _, chunk in indexed]


def build_reranker(settings, llm: object = None) -> Reranker | None:
    """Instantiate the reranker selected by ``RERANKER`` (None = stage off).

    For ``llm`` mode the app's chat client is reused when it is
    OpenAI-compatible; otherwise (e.g. ``LLM_BACKEND=mock``) a client is
    built from the same ``LLM_*`` settings.
    """
    mode = settings.reranker.lower()
    if mode == "none":
        return None
    if mode == "mock":
        return MockReranker()
    if mode == "llm":
        client = (
            llm
            if isinstance(llm, OpenAIChatLLM)
            else OpenAIChatLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_s=settings.llm_timeout_s,
            )
        )
        return LLMReranker(client)
    raise ValueError(f"unknown RERANKER: {settings.reranker!r}")
