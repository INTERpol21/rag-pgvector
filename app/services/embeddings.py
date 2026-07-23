"""Embedding backends behind a minimal ``Embedder`` protocol."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

from app.core.errors import ProviderError

if TYPE_CHECKING:
    from app.core.settings import Settings

# Unicode word tokens, same model as the BM25 leg in app/db/store.py. An
# ASCII-only regex made every non-Latin text hash to the zero vector, so a
# Russian corpus was unsearchable in the offline embedder.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


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
        # Deterministic non-cryptographic bucketing (hash the token into a vector
        # slot); usedforsecurity=False keeps this off the weak-hash security path.
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
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


# Curated synonym clusters for the semantic mock. Every token in a group maps
# to the *same* deterministic basis direction, so members ("car"/"automobile")
# come out near-identical while different groups are near-orthogonal. This is
# still a toy — it only knows the words listed here — but it gives the offline
# stack a taste of the synonym-awareness real embeddings have, which pure
# feature hashing (HashingEmbedder) fundamentally cannot.
_CONCEPT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vehicle", ("car", "cars", "automobile", "automobiles", "vehicle",
                 "vehicles", "sedan", "truck", "motor")),
    ("finance", ("money", "cash", "payment", "payments", "invoice", "invoices",
                 "billing", "cost", "costs", "price", "pricing")),
    ("database", ("postgres", "postgresql", "database", "databases", "sql",
                  "pgvector", "index", "indexes", "query", "queries", "table")),
    ("coffee", ("espresso", "coffee", "latte", "cappuccino", "crema", "grind",
                "brew", "roast", "beans")),
    ("gateway", ("gateway", "http", "request", "requests", "streaming", "token",
                 "tokens", "rate", "limiter", "throttle")),
    ("weather", ("rain", "storm", "sunny", "cloud", "clouds", "forecast",
                 "temperature", "humid", "snow")),
)
# token -> concept key, built once at import time.
_TOKEN_CONCEPT: dict[str, str] = {
    token: concept for concept, tokens in _CONCEPT_GROUPS for token in tokens
}


def _seed_vector(seed: str, dim: int) -> list[float]:
    """Deterministic, roughly-isotropic unit vector derived from ``seed``.

    md5 is streamed (``seed:counter``) into signed floats in ``[-0.5, 0.5]``,
    then L2-normalized. Two different seeds give near-orthogonal vectors; the
    same seed always gives the same vector, across processes.
    """
    vals: list[float] = []
    counter = 0
    while len(vals) < dim:
        # Deterministic non-cryptographic PRNG stream; usedforsecurity=False keeps
        # this off the weak-hash security path (it is not a security primitive).
        digest = hashlib.md5(f"{seed}:{counter}".encode(), usedforsecurity=False).digest()
        for j in range(0, len(digest), 2):
            vals.append(int.from_bytes(digest[j : j + 2], "big") / 65535.0 - 0.5)
        counter += 1
    vec = vals[:dim]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0.0 else vec


class SemanticMockEmbedder:
    """Deterministic *synonym-aware* embedder for offline demos and tests.

    Like :class:`HashingEmbedder` it needs no network and is reproducible, but
    it is a better production stand-in for retrieval: each token contributes its
    concept's fixed direction (see ``_CONCEPT_GROUPS``), so "car" and
    "automobile" embed to almost the same vector while "car" and "espresso" are
    near-orthogonal. Tokens outside the curated vocabulary fall back to a stable
    per-token direction (behaving like the hashing embedder for rare words).

    The per-text vector is the L2-normalized sum of its tokens' directions, so
    the dot product of two embeddings is their cosine similarity — matching the
    ``vector_cosine_ops`` / ``<=>`` retrieval path.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def _direction(self, token: str) -> list[float]:
        # Concept members share a seed ("concept:vehicle"); unknown tokens get
        # their own ("token:<word>"). Cached so repeated tokens are cheap.
        concept = _TOKEN_CONCEPT.get(token)
        seed = f"concept:{concept}" if concept else f"token:{token}"
        vec = self._cache.get(seed)
        if vec is None:
            vec = _seed_vector(seed, self.dim)
            self._cache[seed] = vec
        return vec

    def _embed_one(self, text: str) -> list[float]:
        acc = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            direction = self._direction(token)
            for i, v in enumerate(direction):
                acc[i] += v
        norm = math.sqrt(sum(v * v for v in acc))
        return [v / norm for v in acc] if norm > 0.0 else acc

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


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the embedder selected by ``EMBEDDINGS_BACKEND``."""
    backend = settings.embeddings_backend.lower()
    if backend == "hash":
        return HashingEmbedder(dim=settings.embedding_dim)
    if backend == "semantic":
        return SemanticMockEmbedder(dim=settings.embedding_dim)
    if backend == "openai":
        # dim must come from settings: the store builds its pgvector schema
        # from embedder.dim, so a hard-coded default here would silently size
        # the schema for one model while the API returns another width and
        # every insert would then die on the dimension guard.
        return OpenAIEmbedder(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    raise ValueError(f"unknown EMBEDDINGS_BACKEND: {settings.embeddings_backend!r}")
