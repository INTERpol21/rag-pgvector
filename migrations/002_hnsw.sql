-- ANN index: HNSW over cosine distance (matches the <=> search in app/store.py).
-- Trades a little recall for a lot of speed versus a sequential scan. Unlike
-- IVFFlat it needs no training data, so it can be built on an empty table;
-- prefer IVFFlat (lists ~ sqrt(rows), built after bulk load) only when index
-- build time or memory dominates.

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
