"""Answer synthesis: RAG prompt construction + LLM backends.

The default ``MockLLM`` is fully offline and deterministic; ``OpenAIChatLLM``
talks to any OpenAI-compatible ``/chat/completions`` endpoint — by default
the sibling `llm-gateway <https://github.com/INTERpol21/llm-gateway>`_
on ``http://localhost:8080/v1``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import httpx

from app.errors import ProviderError
from app.store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY "
    "the numbered context blocks provided. Cite the blocks you used inline as "
    "[1], [2], ... If the context does not contain the answer, say you don't "
    "know instead of guessing."
)

NO_CONTEXT_ANSWER = (
    "I don't know — no relevant context was retrieved for this question."
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
    usage: Optional[dict] = None


@runtime_checkable
class LLM(Protocol):
    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult: ...


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}


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

    async def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": build_messages(question, chunks),
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


def build_llm(settings) -> LLM:
    """Instantiate the LLM selected by ``LLM_BACKEND``."""
    backend = settings.llm_backend.lower()
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAIChatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown LLM_BACKEND: {settings.llm_backend!r}")
