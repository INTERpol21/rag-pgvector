-- Tune the HNSW index: higher m / ef_construction trade build time for recall.
-- (pgvector >= 0.8 additionally supports hnsw.iterative_scan for filtered
-- queries; set it at runtime when the build supports it — harmless to omit.)
DROP INDEX IF EXISTS chunks_embedding_hnsw;
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Refresh planner statistics after swapping the index so cost estimates reflect
-- the rebuilt structure (and the source/priority columns from 005) rather than
-- stale pre-migration stats.
ANALYZE chunks;
ANALYZE documents;
