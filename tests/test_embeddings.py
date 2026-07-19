import math

from app.embeddings import HashingEmbedder


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=False))


async def test_deterministic_and_dim():
    e1 = HashingEmbedder(dim=256)
    e2 = HashingEmbedder(dim=256)  # separate instance, separate process-safe hash
    v1 = (await e1.embed(["Postgres stores vectors"]))[0]
    v2 = (await e2.embed(["Postgres stores vectors"]))[0]
    assert v1 == v2
    assert len(v1) == 256 == e1.dim
    # case-insensitive tokenization
    v3 = (await e1.embed(["POSTGRES STORES VECTORS"]))[0]
    assert v3 == v1


async def test_l2_normalized():
    embedder = HashingEmbedder()
    vectors = await embedder.embed(
        ["one", "a longer text with many repeated words words words", ""]
    )
    assert math.isclose(math.sqrt(_dot(vectors[0], vectors[0])), 1.0, rel_tol=1e-9)
    assert math.isclose(math.sqrt(_dot(vectors[1], vectors[1])), 1.0, rel_tol=1e-9)
    # empty text -> zero vector (norm 0), not NaN
    assert all(v == 0.0 for v in vectors[2])


async def test_similar_texts_closer_than_dissimilar():
    embedder = HashingEmbedder()
    query, similar, dissimilar = await embedder.embed(
        [
            "tuning ivfflat lists and probes in pgvector",
            "pgvector ivfflat indexes are tuned via lists and probes",
            "the espresso machine steams milk for a flat white",
        ]
    )
    assert _dot(query, similar) > _dot(query, dissimilar)
