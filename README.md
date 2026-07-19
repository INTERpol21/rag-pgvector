# rag-pgvector

![ci](https://github.com/INTERpol21/rag-pgvector/actions/workflows/ci.yml/badge.svg)

RAG service on FastAPI + Postgres/pgvector: swappable embedding and LLM backends, inline `[n]` citations, and an LLM-as-a-Judge evals harness wired into CI. Runs fully offline by default (in-memory store, hashing embedder, mock LLM); env vars switch it to pgvector and real models.

## What this demonstrates

- The two-pipeline RAG shape (ingest / query) with clean seams between chunking, embedding, storage and synthesis.
- pgvector specifics: `vector(N)` columns, cosine `<=>` search, `ON DELETE CASCADE` chunk lifecycle, IVFFlat indexing notes in `app/store.py`.
- An `Embedder` protocol: deterministic feature hashing offline, OpenAI-compatible client for real retrieval.
- Grounded answers: numbered context blocks, a prompt that forbids answering outside them, `[n]` parsed back into citation objects.
- Evals with metrics (hit_rate@k, citation_presence, judge score) usable as a CI quality gate.

## Architecture

```mermaid
flowchart LR
    subgraph ingest["POST /ingest"]
        A["documents"] --> B["chunk<br/>800 chars / 150 overlap"] --> C["embed<br/>hash 256-d | text-embedding-3-small"]
    end
    subgraph query["POST /query"]
        Q["question"] --> S["embed + hybrid top-k<br/>vector + keyword (RRF)"] --> P["prompt: numbered context,<br/>answer only from it"] --> L["LLM<br/>mock | llm-gateway"] --> R["answer + citations"]
    end
    C --> V[("vector store<br/>memory | pgvector")] --> S
```

## Quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:create_app --factory --port 8081   # offline: no Postgres, no upstream API keys
```

`/ingest`, `/query` and `/stats` require a bearer token (`Authorization: Bearer <key>`); the default accepted key is `demo-key`, override with `RAG_API_KEYS` (comma-separated). Only `/healthz` is unauthenticated.

Full mode: `docker compose up --build` runs the API with `STORE_BACKEND=pgvector` against `pgvector/pgvector:pg16` on :5433 (schema created on startup). For real synthesis, point `LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8080/v1` at the sibling llm-gateway. All knobs: `.env.example`.

## API

```bash
curl -s localhost:8081/ingest -X POST \
  -H 'content-type: application/json' -H 'Authorization: Bearer demo-key' \
  -d '{"documents": [{"id": "pgvector_internals", "title": "pgvector Internals", "text": "..."}]}'

curl -s localhost:8081/query -X POST \
  -H 'content-type: application/json' -H 'Authorization: Bearer demo-key' \
  -d '{"question": "Which pgvector operator matches cosine?", "top_k": 4}'
```

```json
{
  "answer": "... `<=>` — cosine distance, `vector_cosine_ops` [1] ...",
  "citations": [{"document_id": "pgvector_internals", "chunk_id": "pgvector_internals:1",
                 "snippet": "## Distance operators ...", "score": 0.44}],
  "retrieved": [{"chunk_id": "pgvector_internals:1", "ord": 1, "score": 0.44}],
  "usage": {"prompt_tokens": 416, "completion_tokens": 58, "total_tokens": 474}
}
```

`GET /stats` reports backends and document/chunk counts; `GET /healthz` is the liveness probe.

## Evals

```bash
python evals/run_evals.py --min-hit-rate 0.7
```

Runs the full pipeline over `evals/golden.jsonl` (12 questions on the 4-note demo corpus in `data/`) and writes `evals/report.md`. Sample run: hit_rate@4 1.00, citation_presence 1.00, judge_score 3.42/5. The default judge is a deterministic keyword-overlap mock, so CI stays reproducible; `JUDGE_BACKEND=openai` scores with a real model through the gateway. `--min-hit-rate` fails the build on retrieval regressions.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `RAG_API_KEYS` | `demo-key` | comma-separated bearer tokens for `/ingest`, `/query`, `/stats` |
| `STORE_BACKEND` | `memory` | `memory` or `pgvector` (`DATABASE_URL` for the latter) |
| `EMBEDDINGS_BACKEND` | `hash` | `hash` (256-d, deterministic) or `openai` |
| `SEARCH_MODE` | `hybrid` | `hybrid` (RRF of vector + keyword) or `vector` (cosine-only) |
| `RERANKER` | `none` | `none`, `mock` or `llm` |
| `LLM_BACKEND` | `mock` | `mock` or `openai`; `LLM_BASE_URL` defaults to the gateway on :8080 |
| `JUDGE_BACKEND` | `mock` | judge used by the evals harness |

## Notes

- The mock LLM is extractive — it copies question-relevant sentences from the top chunks and cites them. A real model swaps in with the same citation contract.
- Re-ingesting a document id replaces its chunks (FK cascade).

## Testing

26 tests, offline against the in-memory stack: `pip install -r requirements-dev.txt && python -m pytest`; lint with `ruff check app tests evals`.
The 3 `PgVectorStore` integration tests skip unless `DATABASE_URL` points at a live pgvector Postgres (compose db: `postgresql://rag:rag@localhost:5433/rag`).

---

MIT. Portfolio demo — siblings: [llm-gateway](https://github.com/INTERpol21/llm-gateway) · [mcp-tools-server](https://github.com/INTERpol21/mcp-tools-server) · [agent-orchestrator](https://github.com/INTERpol21/agent-orchestrator)
