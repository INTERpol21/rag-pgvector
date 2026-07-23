-- Multilingual keyword retrieval: switch the FTS leg from 'simple' to
-- 'russian'. Postgres's russian config routes Cyrillic words through
-- russian_stem and ASCII words through english_stem, so BOTH languages get
-- stemming ("поиске" matches "поиск", "indexes" matches "index") — 'simple'
-- required exact word forms, which made inflected Russian queries miss.
-- The memory BM25 leg mirrors this with the same Snowball algorithms
-- (app/services/textnorm.py); keep the two in lockstep.
--
-- Trade-off vs 003's rationale: stemming + stopword removal means exact rare
-- FORMS are folded together; genuinely non-word tokens (error codes, model
-- names with digits) are not stemmed and remain searchable as-is.
--
-- The column is GENERATED, so the config cannot be altered in place — drop
-- and recreate (cheap: recomputed from content), then rebuild the GIN index.

DROP INDEX IF EXISTS chunks_content_tsv_gin;

ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv;

ALTER TABLE chunks
    ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED;

CREATE INDEX chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);
