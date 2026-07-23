-- Key-value metadata about the index itself. First use: the fingerprint of
-- the embedder that wrote the vectors, so startup can detect an embedder
-- switch and re-embed instead of silently mixing incompatible vector spaces
-- (content-hash dedup would otherwise keep the old vectors forever).
CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
