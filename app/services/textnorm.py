"""Shared token normalization for the offline retrieval paths.

One tokenizer + stemmer pair used by BOTH the memory BM25 leg and the offline
embedders, so "поиск" and "поиске" (or "index" and "indexes") land on the same
term everywhere. Snowball via the pure-Python ``snowballstemmer`` package —
the SAME algorithms Postgres's ``russian`` FTS config runs (russian_stem for
Cyrillic words, english_stem for ASCII words), which keeps the memory leg and
the pgvector leg (migrations/007) in agreement about what matches.

Script routing is per-token: Cyrillic → Russian stemmer, ASCII letters →
English stemmer, anything else (digits, CJK, mixed) passes through unstemmed —
wrong-language stemming mangles terms worse than no stemming.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC_RE = re.compile(r"^[Ѐ-ӿ]+$")
_ASCII_ALPHA_RE = re.compile(r"^[a-z]+$")


@lru_cache(maxsize=2)
def _stemmer(language: str) -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer(language)


def _stem(token: str) -> str:
    # snowballstemmer ships no type stubs, so stemWord comes back as Any.
    if _CYRILLIC_RE.match(token):
        return str(_stemmer("russian").stemWord(token))
    if _ASCII_ALPHA_RE.match(token):
        return str(_stemmer("english").stemWord(token))
    return token


def normalize_tokens(text: str) -> list[str]:
    """Lowercased, Unicode-tokenized, per-script Snowball-stemmed terms."""
    return [_stem(token) for token in _TOKEN_RE.findall(text.lower())]
