"""Semantic mock embedder: synonym-awareness + the retrieval it unlocks.

The point of ``SemanticMockEmbedder`` over ``HashingEmbedder`` is that synonyms
land near each other in vector space, so a query can retrieve a document that
never repeats the query's exact words — the property real embeddings have and
pure feature-hashing cannot. These tests pin that property and its determinism.
"""

from __future__ import annotations

import math

import pytest

from app.db.store import MemoryVectorStore
from app.services.embeddings import HashingEmbedder, SemanticMockEmbedder


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def test_synonyms_are_close_unrelated_are_far():
    emb = SemanticMockEmbedder(dim=256)
    car, automobile, espresso = await emb.embed(["car", "automobile", "espresso"])
    # same concept cluster -> (near-)identical direction
    assert _cos(car, automobile) > 0.99
    # different clusters -> near-orthogonal
    assert abs(_cos(car, espresso)) < 0.2
    # and the synonym pair is much closer than the unrelated pair
    assert _cos(car, automobile) > _cos(car, espresso) + 0.5


async def test_hashing_embedder_cannot_relate_synonyms():
    # Contrast: the bag-of-words hashing embedder maps synonyms to orthogonal
    # buckets. This is exactly the gap the semantic mock fills.
    car, automobile = await HashingEmbedder(dim=256).embed(["car", "automobile"])
    assert _cos(car, automobile) < 0.2


async def test_vectors_are_normalized_and_dim_honored():
    emb = SemanticMockEmbedder(dim=64)
    vecs = await emb.embed(["pgvector cosine query", "espresso grind dose"])
    for v in vecs:
        assert len(v) == 64
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


async def test_determinism_across_instances():
    a = (await SemanticMockEmbedder(dim=128).embed(["database index query"]))[0]
    b = (await SemanticMockEmbedder(dim=128).embed(["database index query"]))[0]
    assert a == b


def test_rejects_non_positive_dim():
    with pytest.raises(ValueError, match="dim must be positive"):
        SemanticMockEmbedder(dim=0)


async def test_semantic_retrieval_matches_across_synonyms():
    """A car document is retrieved by an 'automobile' query it never mentions."""
    from app.db.store import ChunkRecord, DocumentRecord

    emb = SemanticMockEmbedder(dim=256)
    store = MemoryVectorStore()
    car_vec = (await emb.embed(["cars are fast vehicles with a motor"]))[0]
    coffee_vec = (await emb.embed(["espresso coffee grind and crema"]))[0]
    await store.upsert(
        DocumentRecord(id="car", title="Cars"),
        [ChunkRecord("car:0", "car", 0, "cars are fast vehicles with a motor", car_vec)],
    )
    await store.upsert(
        DocumentRecord(id="coffee", title="Coffee"),
        [ChunkRecord("coffee:0", "coffee", 0, "espresso coffee grind and crema", coffee_vec)],
    )

    query_vec = (await emb.embed(["automobile"]))[0]
    results = await store.search(query_vec, top_k=2)
    assert results[0].document_id == "car"
    assert results[0].score > results[1].score
