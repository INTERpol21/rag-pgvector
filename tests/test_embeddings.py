import math

from app.services.embeddings import HashingEmbedder


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


async def test_non_latin_text_embeds_to_nonzero_vector():
    """Cyrillic text used to hash to the zero vector (ASCII-only tokenizer),
    making a Russian corpus unsearchable in the offline embedder."""
    embedder = HashingEmbedder(dim=32)
    [vec] = await embedder.embed(["Привет мир это заметка"])
    assert any(v != 0.0 for v in vec)

    # Determinism must hold for Unicode input too.
    [again] = await embedder.embed(["Привет мир это заметка"])
    assert vec == again


def test_build_embedder_selects_backends():
    """The factory is string-dispatch on EMBEDDINGS_BACKEND; every branch and
    its wiring matters — the openai branch once dropped EMBEDDING_DIM entirely."""
    from app.core.settings import Settings
    from app.services.embeddings import OpenAIEmbedder, build_embedder

    base = {"embedding_dim": 64, "embedding_model": "mock-small"}

    hashing = build_embedder(Settings(embeddings_backend="hash", **base))
    assert isinstance(hashing, HashingEmbedder) and hashing.dim == 64

    gateway = build_embedder(
        Settings(
            embeddings_backend="gateway",
            llm_base_url="http://gw:8080/v1",
            llm_api_key="k",
            **base,
        )
    )
    assert isinstance(gateway, OpenAIEmbedder)
    # The gateway branch must reuse the LLM connection settings — not OPENAI_*.
    assert gateway.base_url == "http://gw:8080/v1"
    assert gateway.api_key == "k"
    assert gateway.model == "mock-small"
    assert gateway.dim == 64

    openai = build_embedder(
        Settings(embeddings_backend="openai", openai_base_url="http://api/v1", **base)
    )
    assert isinstance(openai, OpenAIEmbedder)
    assert openai.base_url == "http://api/v1"
    assert openai.dim == 64

    import pytest

    with pytest.raises(ValueError):
        build_embedder(Settings(embeddings_backend="nope", **base))
