-- Index the FK column so re-ingest (DELETE FROM chunks WHERE document_id = $1)
-- and cascade deletes use an index scan instead of a sequential scan of chunks.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
