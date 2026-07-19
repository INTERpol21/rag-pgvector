"""Answer synthesis: RAG prompt construction + LLM backends.

The default ``MockLLM`` is fully offline and deterministic; ``OpenAIChatLLM``
talks to any OpenAI-compatible ``/chat/completions`` endpoint — by default
the sibling `llm-gateway <https://github.com/INTERpol21/llm-gateway>`_
on ``http://localhost:8080/v1``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from app.core.errors import ProviderError
from app.db.store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY "
    "the numbered context blocks provided. Cite the blocks you used inline as "
    "[1], [2], ... If the context does not contain the answer, say you don't "
    "know instead of guessing."
)

NO_CONTEXT_ANSWER = (
    "I don't know — no relevant context was retrieved for this question."
)

# Distinct from NO_CONTEXT_ANSWER: chunks *were* retrieved but none of them are
# relevant to the question. A production RAG system abstains here rather than
# summarising an off-topic passage; the grounded mock emulates that guardrail.
NOT_IN_SOURCES_ANSWER = (
    "I couldn't find an answer to that in the provided sources."
)


def format_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered context blocks (1-based)."""
    blocks = [
        f"[{i}] (from \"{c.title}\")\n{c.content.strip()}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[ScoredChunk]) -> list[dict]:
    """Build the chat messages for RAG synthesis."""
    user = (
        f"Context:\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer from the context above, citing sources as [n]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class LLMResult:
    answer: str
    usage: dict | None = None


@runtime_checkable
class LLM(Protocol):
    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult: ...


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Common function words that carry no topical signal. Grounding relevance must
# ignore these, otherwise a shared "the"/"does" between an off-topic question and
# any passage would count as a match and defeat honest abstention.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "are", "was", "with", "that", "this", "from",
        "has", "have", "had", "use", "uses", "used", "what", "which", "does",
        "did", "how", "why", "who", "whom", "where", "when", "into", "than",
        "then", "its", "you", "your", "our", "their", "them", "they", "she",
        "his", "her", "not", "but", "all", "any", "can", "will", "may", "about",
        "over", "under", "such", "some", "these", "those", "there", "here",
    }
)


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}


def _content_tokens(text: str) -> set[str]:
    """Topical tokens only: drop short words and common stopwords."""
    return _tokens(text) - _STOPWORDS


def _best_sentence(question: str, content: str, max_chars: int = 280) -> str:
    """Pick the sentence of ``content`` sharing the most words with the question."""
    flat = " ".join(content.split())  # collapse whitespace/newlines
    sentences = [s for s in _SENTENCE_END_RE.split(flat) if s.strip()]
    q_tokens = _tokens(question)
    # max() keeps the FIRST best on ties -> deterministic
    best = max(sentences, key=lambda s: len(_tokens(s) & q_tokens)) if sentences else flat
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(" ", 1)[0] + "…"
    return best


def _approx_tokens(text: str) -> int:
    """Rough subword-token estimate: ~1.3 tokens per whitespace word.

    Real BPE tokenizers split on subwords and punctuation, so a bare word count
    understates usage. The 1.3 factor keeps the mock's ``usage`` numbers in the
    same ballpark as a production tokenizer without pulling in ``tiktoken``.
    """
    words = len(text.split())
    return math.ceil(words * 1.3) if words else 0


def _prompt_tokens(messages: Sequence[dict]) -> int:
    return _approx_tokens("\n".join(m["content"] for m in messages))


class MockLLM:
    """Deterministic offline stand-in for a chat model.

    Does no generation. It answers extractively: from each top retrieved
    chunk it copies the sentence with the highest word overlap with the
    question and appends an [n] citation — enough to exercise the full RAG
    loop (prompting, citation extraction, evals) without a model or network.
    """

    max_chunks = 2

    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult:
        # Build (and count) the real prompt so usage numbers are plausible.
        messages = build_messages(question, chunks)
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        if not chunks:
            return LLMResult(
                answer=NO_CONTEXT_ANSWER,
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": 0},
            )
        parts = [
            f"{_best_sentence(question, chunk.content).rstrip('.')} [{i}]."
            for i, chunk in enumerate(chunks[: self.max_chunks], start=1)
        ]
        answer = " ".join(parts)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(answer.split()),
            "total_tokens": prompt_tokens + len(answer.split()),
        }
        return LLMResult(answer=answer, usage=usage)


class GroundedMockLLM:
    """Production-emulating offline LLM: grounded extractive RAG with guardrails.

    Where :class:`MockLLM` always answers from the top chunks, this stand-in
    reproduces the two behaviours a real retrieval-augmented model is judged on,
    so tests can exercise them without a model or network:

    * **Grounding.** It answers only from retrieved chunks that actually share
      wording with the question, and cites each *used* chunk with a ``[n]`` that
      is always a valid 1-based position in the retrieved list. It can never
      emit a hallucinated citation, and chunks it did not use are never cited —
      so :func:`app.services.citations.extract_citations` resolves every ``[n]``.

    * **Honest abstention.** If chunks were retrieved but none are relevant to
      the question (they share no meaningful content word), it returns the
      distinct :data:`NOT_IN_SOURCES_ANSWER` instead of summarising an off-topic
      passage. An empty retrieval yields :data:`NO_CONTEXT_ANSWER`, as before.

    ``usage`` approximates a real tokenizer (~1.3 subword tokens per word) and
    always carries ``prompt_tokens``/``completion_tokens``/``total_tokens``.
    Fully deterministic and offline; select it with ``LLM_BACKEND=grounded``.
    """

    # How many relevant chunks to weave into the answer (one [n] citation each).
    max_chunks = 3
    # Minimum shared content words for a chunk to count as relevant to the
    # question. 1 keeps recall high while still rejecting genuinely off-topic hits.
    min_overlap = 1

    def _relevant(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> list[tuple[int, ScoredChunk]]:
        """Retrieved chunks (with their 1-based index) that overlap the question.

        Order is preserved from ``chunks`` so the highest-ranked relevant chunk
        is cited first; the index is the chunk's position in the retrieved list,
        which is exactly what the citation resolver expects.
        """
        q_tokens = _content_tokens(question)
        hits: list[tuple[int, ScoredChunk]] = []
        for i, chunk in enumerate(chunks, start=1):
            if len(q_tokens & _tokens(chunk.content)) >= self.min_overlap:
                hits.append((i, chunk))
        return hits

    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult:
        messages = build_messages(question, chunks)
        prompt_tokens = _prompt_tokens(messages)

        def _result(answer: str) -> LLMResult:
            completion_tokens = _approx_tokens(answer)
            return LLMResult(
                answer=answer,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

        if not chunks:
            return _result(NO_CONTEXT_ANSWER)
        relevant = self._relevant(question, chunks)[: self.max_chunks]
        if not relevant:
            # Retrieval returned rows, but none answer the question -> abstain.
            return _result(NOT_IN_SOURCES_ANSWER)
        parts = [
            f"{_best_sentence(question, chunk.content).rstrip('.')} [{i}]."
            for i, chunk in relevant
        ]
        return _result(" ".join(parts))


class OpenAIChatLLM:
    """Synthesis via an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    async def complete(self, messages: list[dict]) -> LLMResult:
        """Raw chat completion for arbitrary messages (also used by the
        LLM reranker, which needs a scoring prompt rather than RAG synthesis)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            # trust_env=False: the synthesis endpoint is an internal service
            # (usually the sibling llm-gateway); system proxies must not reroute it.
            async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"LLM provider failed: {exc}") from exc
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"LLM returned malformed response: {data!r}") from exc
        return LLMResult(answer=answer or "", usage=data.get("usage"))

    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult:
        return await self.complete(build_messages(question, chunks))


def build_llm(settings) -> LLM:
    """Instantiate the LLM selected by ``LLM_BACKEND``."""
    backend = settings.llm_backend.lower()
    if backend == "mock":
        return MockLLM()
    if backend == "grounded":
        return GroundedMockLLM()
    if backend == "openai":
        return OpenAIChatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown LLM_BACKEND: {settings.llm_backend!r}")
