# pgvector Internals

pgvector is a Postgres extension that adds a `vector` column type and
similarity operators, turning a boring, reliable relational database into a
perfectly good vector store for small-to-medium corpora.

## The vector type

`CREATE EXTENSION vector;` enables columns like `embedding vector(1536)`.
Vectors are stored as arrays of 4-byte floats with a fixed, declared
dimension; inserting a vector of the wrong dimension is an error. Because it
is a regular column, embeddings live next to their metadata, share
transactions with it, and are backed up by the same `pg_dump`.

## Distance operators

pgvector ships three distance operators, each with a matching index opclass:

- `<->` — Euclidean (L2) distance, `vector_l2_ops`
- `<#>` — negative inner product, `vector_ip_ops`
- `<=>` — cosine distance, `vector_cosine_ops`

Cosine distance is `1 - cosine similarity`, so a typical search is
`ORDER BY embedding <=> $1 LIMIT k` and the similarity reported to users is
`1 - distance`. For normalized embeddings, inner product and cosine give the
same ranking, and inner product is slightly cheaper.

## Exact vs approximate search

Without an index, every query is a sequential scan: exact results, O(rows)
cost. That is completely fine up to roughly a hundred thousand vectors. Past
that, pgvector offers two approximate-nearest-neighbor (ANN) index types.

### IVFFlat

`CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH
(lists = 100);` clusters vectors into `lists` cells (k-means at build time)
and searches only the closest `probes` cells per query. Build the index
AFTER loading data, pick `lists` around sqrt(row_count), and raise
`SET ivfflat.probes` to trade latency for recall. IVFFlat is small and fast
to build, but recall degrades if data drifts far from the original clusters.

### HNSW

`CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH
(m = 16, ef_construction = 64);` builds a multi-layer proximity graph.
Queries greedily descend the graph, controlled at runtime by
`SET hnsw.ef_search`. HNSW gives better recall/latency trade-offs than
IVFFlat and does not need a representative dataset at build time, at the
price of slower builds and more memory.

## Operational notes

ANN queries return approximate results — expect recall below 1.0 and test
it against exact scans on your own data. Combine vector predicates with
regular `WHERE` filters carefully: post-filtering after an ANN scan can
return fewer than k rows. Keep an eye on `maintenance_work_mem` during
index builds, and remember that `VACUUM` matters for vector tables too.
