import random

import pytest

from app.services.chunking import chunk_text

SIZE = 200
OVERLAP = 40


def make_unique_text(n_words: int = 400) -> str:
    """Every token unique so chunk positions are unambiguous."""
    words = [f"w{i:04d}" for i in range(n_words)]
    # sprinkle paragraph and sentence boundaries
    parts = []
    for i, w in enumerate(words):
        parts.append(w)
        if i % 37 == 36:
            parts.append("\n\n")
        elif i % 11 == 10:
            parts.append(". ")
        else:
            parts.append(" ")
    return "".join(parts).strip()


def test_short_doc_single_chunk():
    assert chunk_text("hello world", chunk_size=SIZE, chunk_overlap=OVERLAP) == [
        "hello world"
    ]
    exact = "x" * SIZE  # exact boundary: still a single chunk
    assert chunk_text(exact, chunk_size=SIZE, chunk_overlap=OVERLAP) == [exact]


def test_no_content_loss_and_exact_overlap():
    text = make_unique_text()
    chunks = chunk_text(text, chunk_size=SIZE, chunk_overlap=OVERLAP)
    assert len(chunks) > 2
    assert all(len(c) <= SIZE for c in chunks)
    # every chunk is a real substring
    assert all(c in text for c in chunks)
    # consecutive chunks share exactly OVERLAP characters
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a[-OVERLAP:] == b[:OVERLAP]
    # dropping each chunk's overlap prefix reconstructs the original text
    rebuilt = chunks[0] + "".join(c[OVERLAP:] for c in chunks[1:])
    assert rebuilt == text


def test_separator_preference():
    text = "A" * 150 + "\n\n" + "B" * 120 + "\n\n" + "C" * 150
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    # the first cut should land on the paragraph break, not mid-word
    assert chunks[0].endswith("\n\n")

    sentences = "First sentence about apples. " * 20
    chunks = chunk_text(sentences.strip(), chunk_size=120, chunk_overlap=20)
    # every cut lands on a sentence boundary
    assert all(c.rstrip().endswith(".") for c in chunks)
    assert chunks[0].endswith(". ")


def test_edge_cases():
    assert chunk_text("", chunk_size=SIZE, chunk_overlap=OVERLAP) == []
    assert chunk_text("   \n\n  ", chunk_size=SIZE, chunk_overlap=OVERLAP) == []
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, chunk_overlap=100)  # overlap >= size
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, chunk_overlap=0)
    # text with no separators at all: hard cuts, still lossless
    blob = "z" * 950
    chunks = chunk_text(blob, chunk_size=400, chunk_overlap=100)
    rebuilt = chunks[0] + "".join(c[100:] for c in chunks[1:])
    assert rebuilt == blob


def test_property_fuzz_random_unicode_and_bounds():
    """200 random unicode texts x random valid (size, overlap): invariants hold."""
    rng = random.Random(1234)
    alphabet = "abc 0123 .\n\té中\U0001F600‏‎"  # incl. emoji + RTL/LTR marks

    def non_ws(s: str) -> str:
        return "".join(s.split())

    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 500)))
        size = rng.randint(1, 200)
        overlap = rng.randint(0, size - 1)  # valid range: 0 <= overlap < size
        chunks = chunk_text(text, size, overlap)
        # size bound: no chunk exceeds chunk_size
        assert all(len(c) <= size for c in chunks)
        # every chunk is a contiguous substring of the input (no fabrication)
        assert all(c in text for c in chunks)
        # no-loss with overlap==0: chunks tile the text, only whitespace-only
        # slices are dropped, so no non-whitespace character is ever lost.
        if overlap == 0:
            assert non_ws("".join(chunks)) == non_ws(text)


def test_property_fuzz_terminates_on_degenerate_sizes():
    """size=1/overlap=0 on a long blob terminates and stays within bounds."""
    text = "a" * 1000 + " " + "b" * 1000
    chunks = chunk_text(text, chunk_size=1, chunk_overlap=0)
    assert max(len(c) for c in chunks) == 1
    # 2000 non-space characters, each its own chunk (spaces dropped as blank)
    assert len(chunks) == 2000
