# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-07-23

### Added
- Multilingual retrieval. Both keyword legs now stem word forms with the same
  Snowball algorithms: the Postgres FTS column and query moved from `simple`
  to the `russian` config (Cyrillic via russian_stem, ASCII via english_stem —
  migrations/007), and the memory BM25 leg plus the offline embedders share a
  new tokenizer (`app/services/textnorm.py`, pure-Python `snowballstemmer`).
  "векторный поиск" now finds "векторном поиске" in memory, in pgvector, and
  in the hash-embedded vector leg. Non-word tokens (error codes, digit-bearing
  names) pass through unstemmed and stay exact-searchable.
- A Russian note in the demo corpus plus Russian golden questions, so the eval
  harness exercises Cyrillic retrieval end to end.

### Changed
- Offline embedder vectors changed (tokens are now stemmed): a pgvector store
  populated with old hash/semantic-mock embeddings should be re-ingested. The
  openai/gateway embedding backends are unaffected.

## [1.1.0] - 2026-07-23

### Added
- `EMBEDDINGS_BACKEND=gateway`: fetch vectors through the sibling llm-gateway's
  `/v1/embeddings` instead of embedding in-process — one platform entrypoint,
  one usage/cost ledger, retries and fallbacks inherited from the gateway.
  Reuses the `LLM_BASE_URL`/`LLM_API_KEY` connection; `EMBEDDING_MODEL` must
  name a gateway route (offline: `mock-small`).

### Changed
- CI runs the 11 pgvector integration tests against a real Postgres service
  container (87 tests, zero skips) and fails loudly if they ever silently skip
  again.

## [1.0.0] - 2026-07-21

First tagged release. A RAG service on FastAPI + Postgres/pgvector with hybrid
retrieval and an evals harness in CI.

### Added
- `POST /v1/ingest` and `/v1/ingest/file` (md/txt/pdf/docx, 10 MB cap), with
  content-hash idempotency so unchanged documents are skipped.
- `POST /v1/query` returning an answer with inline `[n]` citations, and
  `GET /v1/stats` reporting the active backends.
- Hybrid retrieval: cosine vector search fused with a keyword leg (Postgres FTS,
  or Okapi BM25 in the memory store) via RRF; optional reranker.
- Local-first provenance: `source`/`priority`/`owner` tags with strict and
  priority search modes.
- pgvector schema under `migrations/`: HNSW index, generated tsvector + GIN,
  cascade deletes, startup dimension guard.
- Swappable embedding and LLM backends; offline defaults (memory store, hashing
  embedder, mock LLM) need no Postgres and no API keys.
- LLM-as-a-Judge evals harness gated in CI on hit-rate, plus a promptfoo
  OWASP-LLM suite on the synthesis boundary.

### Notes
- The offline embedder and mock LLM are deterministic demo backends behind
  interfaces, not production components.
- pgvector integration tests are skipped unless a live `DATABASE_URL` is set.

[1.0.0]: https://github.com/INTERpol21/rag-pgvector/releases/tag/v1.0.0
