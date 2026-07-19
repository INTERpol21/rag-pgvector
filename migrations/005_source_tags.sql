-- Local-first provenance on documents: your own data (source='local') gets a
-- high priority so it outranks web/other in retrieval. Columns default in a
-- backward-compatible way so pre-existing rows become local/authoritative.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source   text NOT NULL DEFAULT 'local';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS priority int  NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner    text;

-- B-tree index on the provenance filter column (strict "only my data" queries).
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
