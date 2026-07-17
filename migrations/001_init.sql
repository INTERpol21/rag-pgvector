-- Base schema: pgvector extension, documents and chunks.
-- The {dim} placeholder is replaced with the embedding dimension at apply time.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           text PRIMARY KEY,
    title        text NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    content_hash text NOT NULL DEFAULT '',
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Upgrade path for databases created before the migrations framework
-- (their documents table exists without content_hash).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         int  NOT NULL,
    content     text NOT NULL,
    embedding   vector({dim}) NOT NULL
);
