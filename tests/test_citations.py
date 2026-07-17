from app.citations import extract_citations, extract_reference_indices
from app.store import ScoredChunk


def _scored(i: int, doc: str = "doc") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{doc}:{i}",
        document_id=doc,
        title=doc.title(),
        content=f"content of chunk {i} " * 20,
        ord=i,
        score=1.0 - i * 0.1,
    )


RETRIEVED = [_scored(0, "alpha"), _scored(1, "beta"), _scored(2, "gamma")]


def test_extraction_maps_to_chunks():
    answer = "Fact one [1]. Fact two comes from elsewhere [3]."
    citations = extract_citations(answer, RETRIEVED)
    assert [c.chunk_id for c in citations] == ["alpha:0", "gamma:2"]
    assert citations[0].document_id == "alpha"
    assert citations[0].title == "Alpha"
    assert citations[0].score == 1.0
    assert 0 < len(citations[0].snippet) <= 201  # trimmed snippet + ellipsis


def test_missing_and_out_of_range_refs():
    assert extract_citations("No references here.", RETRIEVED) == []
    # [0] and [9] do not resolve to retrieved chunks and are dropped
    citations = extract_citations("Bogus [0] and hallucinated [9] but real [2].", RETRIEVED)
    assert [c.chunk_id for c in citations] == ["beta:1"]
    # empty retrieval -> nothing to cite
    assert extract_citations("Claims [1] things.", []) == []


def test_dedup_preserves_first_appearance_order():
    answer = "B first [2], then A [1], then B again [2] and A again [1]."
    assert extract_reference_indices(answer) == [2, 1]
    citations = extract_citations(answer, RETRIEVED)
    assert [c.chunk_id for c in citations] == ["beta:1", "alpha:0"]


def test_hostile_reference_shapes_are_pinned():
    # [0] is out of range (1-based) and dropped
    assert extract_reference_indices("ref [0] only") == [0]
    assert extract_citations("ref [0] only", RETRIEVED) == []
    # a wildly out-of-range index is dropped, not an IndexError
    assert extract_citations("way out [99]", RETRIEVED) == []
    # adjacency and repetition: dedup, first-appearance order, no crash
    assert extract_reference_indices("[1][2][1]") == [1, 2]
    assert [c.chunk_id for c in extract_citations("[1][2][1]", RETRIEVED)] == [
        "alpha:0",
        "beta:1",
    ]
    # whitespace inside the brackets is NOT a citation (strict [\d+] shape)
    assert extract_reference_indices("spaced [1 ] and [ 1]") == []
    assert extract_citations("spaced [1 ] and [ 1]", RETRIEVED) == []
    # zero-padded index parses as its integer value
    assert extract_reference_indices("[01]") == [1]
    assert [c.chunk_id for c in extract_citations("[01]", RETRIEVED)] == ["alpha:0"]
