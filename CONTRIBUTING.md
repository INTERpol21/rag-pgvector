# Contributing — rag-pgvector

RAG service (:8081): ingest → chunk → embed → vector search → cited synthesis.

## Setup

```bash
make install-dev      # uv sync (runtime + dev)
make run              # uvicorn on :8081 (offline defaults: memory store, mock LLM)
```

`uv` is the source of truth; `requirements*.txt` are exported from `uv.lock`
(`make lock` after editing `pyproject.toml`, and commit the diff).

## Gates (all must be green before a PR)

```bash
make lint             # ruff
make typecheck        # mypy (strict)
make test             # pytest, offline
```

Integration tests hit a real database and are **skipped** unless `DATABASE_URL`
is set. Run them against a live Postgres (pgvector ≥ 0.8):

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/db make test
```

CI runs ruff + mypy + pytest, plus pip-audit (dependency CVEs), bandit (SAST) and
CodeQL. A prompt-injection (OWASP-LLM) eval runs via promptfoo — see
`evals/promptfoo/` and the `llm-eval` CI job.

## Layout (layered skeleton — shared across the four backends)

```
app/
  main.py            # FastAPI factory + lifespan
  core/              # settings · security · errors · logging · middleware
  api/routes/*.py    # HTTP only: route -> service; deps.py for shared deps
  schemas/           # pydantic DTOs
  services/          # business logic (framework-free, testable)
  db/                # store (memory | pgvector), migrations/, startup dim-guard
```

- The API is served under `/v1` (only `/healthz` is unversioned).
- HTTP handlers stay thin; put logic in `services/` so it is unit-testable
  without FastAPI (inject fakes through the app factory).

## Conventions

- **Types:** no `dict[str, Any]` for structured data — use named pydantic
  models / `TypedDict`. mypy strict must pass.
- **Security (untrusted context):** retrieved chunks are data, never
  instructions. They go into fenced, defanged blocks (`services/llm.py`); the
  system prompt forbids obeying them. Add a regression test for any change here.
- **DB changes:** a new `WHERE`/`JOIN` column needs an index; a new embedder
  dimension must keep the startup dim-guard honest; re-ingest must stay
  idempotent (content-hash dedup).
- **Tests:** behaviour through the public surface; deterministic/offline (stub
  time/sleep); one behaviour per test; a regression test for every fix. Don't
  test trivial glue (DTOs, framework wiring).

## Commits & branches

Small, focused commits with a clear subject line. Develop on a feature branch
(`claude/<topic>` in this project) and open a PR against the default branch; keep
gates green.
