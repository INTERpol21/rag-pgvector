"""The shared tokenizer+stemmer both retrieval legs and the offline embedders use."""

from app.services.textnorm import normalize_tokens


def test_russian_word_forms_share_a_stem():
    assert normalize_tokens("векторный поиск") == normalize_tokens("векторном поиске")


def test_english_word_forms_share_a_stem():
    assert normalize_tokens("index queries") == normalize_tokens("indexes query")


def test_mixed_script_routes_per_token():
    # Cyrillic through russian_stem, ASCII through english_stem, in one text.
    assert normalize_tokens("поиске indexes") == normalize_tokens("поиск index")


def test_non_words_pass_through_unstemmed():
    # Error codes / digit-bearing tokens must stay exact-searchable.
    assert normalize_tokens("E1234 v2 http2") == ["e1234", "v2", "http2"]


def test_matches_postgres_russian_config_stems():
    """The stems must equal what to_tsvector('russian', ...) produces — both
    sides run the same Snowball algorithms; drift silently splits the legs."""
    assert normalize_tokens("векторном поиске indexes searching") == [
        "векторн", "поиск", "index", "search",
    ]
