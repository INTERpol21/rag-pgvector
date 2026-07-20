#!/usr/bin/env bash
# Run the OWASP-LLM promptfoo eval against a local rag service.
#
# Starts the service (offline: memory store + grounded mock LLM), seeds the
# corpus so retrieval has context, runs promptfoo, then tears the service down.
# Offline and deterministic — no network, no API keys.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
BASE=${RAG_BASE_URL:-http://localhost:8081}
KEY=${RAG_API_KEY:-demo-key}

echo "→ starting rag (offline defaults) on :8081"
uv run uvicorn app.main:create_app --factory --port 8081 --log-level warning &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for readiness.
for _ in $(seq 1 40); do
  curl -sf "$BASE/healthz" >/dev/null 2>&1 && break
  sleep 0.25
done

echo "→ seeding corpus"
curl -sf -X POST "$BASE/v1/ingest" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  --data-binary @evals/promptfoo/corpus.json >/dev/null

echo "→ running promptfoo eval"
npx -y promptfoo@latest eval -c evals/promptfoo/promptfooconfig.yaml
