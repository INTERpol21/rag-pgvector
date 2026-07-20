# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-20

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

[0.1.0]: https://github.com/INTERpol21/rag-pgvector/releases/tag/v0.1.0
