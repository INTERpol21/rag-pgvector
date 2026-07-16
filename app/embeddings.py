"""Embedding backends behind a minimal ``Embedder`` protocol."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence, runtime_checkable

import httpx

from app.errors import ProviderError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn a batch of texts into fixed-size vectors."""

    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector of length ``dim`` per input text."""
        ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder for offline demos and tests.

    Lowercased ASCII word tokens are hashed (md5, stable across processes)
    into ``dim`` buckets; the term-count vector is L2-normalized, so the dot
    product of two embeddings equals their cosine similarity.

    This is bag-of-words, not semantics: "car" and "automobile" come out
    orthogonal. It exists so the whole pipeline (chunk -> embed -> search ->
    cite) runs with zero network access; swap in ``OpenAIEmbedder`` for real
    retrieval quality.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            vec[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Embeddings via an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout_s = timeout_s

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()["data"]
        except httpx.HTTPError as exc:  # timeouts, connect errors, 4xx/5xx
            raise ProviderError(f"embeddings provider failed: {exc}") from exc
        # The API may return items out of order; sort by index to be safe.
        return [item["embedding"] for item in sorted(data, key=lambda i: i["index"])]


def build_embedder(settings) -> Embedder:
    """Instantiate the embedder selected by ``EMBEDDINGS_BACKEND``."""
    backend = settings.embeddings_backend.lower()
    if backend == "hash":
        return HashingEmbedder(dim=settings.embedding_dim)
    if backend == "openai":
        return OpenAIEmbedder(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    raise ValueError(f"unknown EMBEDDINGS_BACKEND: {settings.embeddings_backend!r}")
