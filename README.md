# rag-pgvector

Retrieval-Augmented Generation service: **FastAPI + Postgres/pgvector**, with
swappable embeddings/LLM backends, inline `[n]` citations and an
**LLM-as-a-Judge evals harness** that runs in CI.

The whole pipeline runs **fully offline by default** (in-memory vector store,
deterministic hashing embedder, mock LLM) and switches to
pgvector + real models via environment variables — no code changes.

## Architecture

**Ingest pipeline** — `POST /ingest`:

```mermaid
flowchart LR
    A["POST /ingest<br/>documents"] --> B["Recursive character chunker<br/>800 chars / 150 overlap"]
    B --> C["Embedder<br/>hash 256-d or text-embedding-3-small"]
    C --> D[("Vector store<br/>memory or pgvector")]
    D --> E["documents + chunks upserted<br/>re-ingest replaces chunks (FK cascade)"]
```

**Query pipeline** — `POST /query`:

```mermaid
flowchart LR
    Q["POST /query<br/>question, top_k"] --> E1["Embed question"]
    E1 --> S["Cosine search<br/>pgvector <=> or in-memory brute force"]
    S --> P["RAG prompt: numbered context blocks<br/>answer only from context, cite as [n]"]
    P --> L["LLM synthesis<br/>mock or llm-gateway"]
    L --> C["Citation extractor<br/>[n] -> chunk -> document"]
    C --> R["answer + citations<br/>+ retrieved + usage"]
```

## What this demonstrates

- **RAG design** — the classic two-pipeline shape (write path / read path)
  with clean seams between chunking, embedding, storage and synthesis.
- **pgvector schema & queries** — `vector(N)` columns, cosine `<=>`
  ordered search, `ON DELETE CASCADE` chunk lifecycle, and a documented
  IVFFlat indexing strategy for scale (`app/store.py`).
- **Embeddings abstraction** — an `Embedder` protocol with a deterministic
  feature-hashing implementation for offline use and an OpenAI-compatible
  client for real semantic retrieval.
- **Grounded answers with citations** — numbered context blocks, a prompt
  that forbids answering outside the context, and `[n]` references parsed
  back into `{document_id, chunk_id, snippet, score}` objects.
- **LLM-as-a-Judge evals** — a golden set + metrics (hit_rate@k,
  citation_presence, 1-5 judge score) with a deterministic mock judge for CI
  and a real judge through the [llm-gateway](https://github.com/INTERpol21/llm-gateway).

## Quickstart (offline — runs instantly)

```bash
pip install -r requirements.txt
uvicorn app.main:create_app --factory --port 8081
```

No Postgres, no API keys: memory store + hashing embedder + mock LLM.

### Full mode (pgvector + real LLM)

```bash
docker compose up --build   # api on :8081, pgvector/pgvector:pg16 on :5433
```

The API container runs with `STORE_BACKEND=pgvector`; the schema (extension,
tables) is created on startup. To synthesize with a real model, point the
service at the sibling [llm-gateway](https://github.com/INTERpol21/llm-gateway):

```bash
LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8080/v1 \
LLM_API_KEY=demo-key LLM_MODEL=gpt-4o-mini \
uvicorn app.main:create_app --factory --port 8081
```

All knobs live in [.env.example](.env.example).

## API

### Ingest documents

```bash
curl -s localhost:8081/ingest -X POST -H 'content-type: application/json' -d '{
  "documents": [
    {"id": "pgvector_internals", "title": "pgvector Internals",
     "text": "pgvector is a Postgres extension that adds a vector column type...",
     "metadata": {"source": "docs"}}
  ]
}'
```

```json
{"document_ids": ["pgvector_internals"], "chunks_indexed": 6}
```

### Ask a question

```bash
curl -s localhost:8081/query -X POST -H 'content-type: application/json' -d '{
  "question": "Which pgvector distance operators exist and which opclass matches cosine?",
  "top_k": 4
}'
```

```json
{
  "answer": "pgvector ships three distance operators, each with a matching index opclass: `<->` — Euclidean (L2) distance, `vector_l2_ops`; `<#>` — negative inner product, `vector_ip_ops`; `<=>` — cosine distance, `vector_cosine_ops` [1]. ...",
  "citations": [
    {
      "document_id": "pgvector_internals",
      "title": "pgvector Internals",
      "chunk_id": "pgvector_internals:1",
      "snippet": "## Distance operators pgvector ships three distance operators, each with a matching index opclass: - `<->` — Euclidean (L2) distance…",
      "score": 0.44
    }
  ],
  "retrieved": [
    {"chunk_id": "pgvector_internals:1", "document_id": "pgvector_internals",
     "title": "pgvector Internals", "ord": 1, "score": 0.44}
  ],
  "usage": {"prompt_tokens": 416, "completion_tokens": 58, "total_tokens": 474}
}
```

(With the default `LLM_BACKEND=mock` the answer is *extractive* — the mock
copies the question-relevant sentences from the top chunks and cites them.
Swap in a real model for actual synthesis; the citation contract is identical.)

### Stats & health

```bash
curl -s localhost:8081/stats
# {"backend": "memory", "documents": 4, "chunks": 22,
#  "embeddings_backend": "hash", "llm_backend": "mock", "embedding_dim": 256}
curl -s localhost:8081/healthz
# {"status": "ok"}
```

## Evals

`data/` ships a small technical corpus — four engineering notes on the stack
this portfolio covers (LLM gateways, RAG architecture, pgvector internals,
MCP). `evals/golden.jsonl` holds 12 question/reference/expected-document
triples over it.

```bash
python evals/run_evals.py --min-hit-rate 0.7
```

Builds a fresh in-memory index, runs the *full* pipeline per item and writes
`evals/report.md` (gitignored). Sample run:

| metric | value |
|---|---|
| hit_rate@4 | 1.00 |
| citation_presence | 1.00 |
| judge_score (1-5) | 3.42 |

The default judge is a deterministic keyword-overlap `MockJudge`, so evals
are reproducible offline and CI-friendly; `JUDGE_BACKEND=openai` scores
answers with a real model through the gateway.
`--min-hit-rate` turns the harness into a CI quality gate: retrieval
regressions fail the build.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
ruff check app tests evals
```

Unit and e2e tests run against the in-memory stack — no network, no database.
The `PgVectorStore` integration test is skipped unless `DATABASE_URL` points
at a live Postgres with pgvector (e.g. the docker-compose `db` on
`postgresql://rag:rag@localhost:5433/rag`).

## Project layout

```
app/
  chunking.py     # recursive character splitter (pure, lossless, overlap-exact)
  embeddings.py   # Embedder protocol: HashingEmbedder | OpenAIEmbedder
  store.py        # VectorStore protocol: MemoryVectorStore | PgVectorStore
  llm.py          # RAG prompt + MockLLM | OpenAIChatLLM (via llm-gateway)
  citations.py    # [n] -> retrieved chunk -> citation objects
  main.py         # create_app() factory, DI via app.state, error handlers
  settings.py     # pydantic-settings
data/             # demo corpus: 4 engineering notes (gateway, RAG, pgvector, MCP)
evals/            # golden.jsonl, run_evals.py, report.md
tests/            # chunker, embedder, store, citations, API e2e, evals smoke
```

> **Note:** this is a portfolio demo project — intentionally compact,
> dependency-light and offline-first. The seams (protocols for embedder,
> store, LLM, judge) are where a production system would plug in real
> infrastructure.

---

**Part of the AI Platform Portfolio:**
[llm-gateway](https://github.com/INTERpol21/llm-gateway) ·
**rag-pgvector** ·
[mcp-tools-server](https://github.com/INTERpol21/mcp-tools-server) ·
[agent-orchestrator](https://github.com/INTERpol21/agent-orchestrator)

MIT © 2026 Anton Zhuravlev
