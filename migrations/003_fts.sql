-- Full-text search side of hybrid retrieval: a generated tsvector column over
-- chunk content plus a GIN index. The 'simple' config skips stemming and
-- stopword removal so exact rare terms (error codes, product names) remain
-- searchable — that is precisely what the BM25/FTS leg is for.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);
