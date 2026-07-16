# RAG Architecture

Retrieval-Augmented Generation (RAG) grounds an LLM's answer in documents
retrieved at query time. Instead of fine-tuning knowledge into weights, the
system fetches relevant passages and instructs the model to answer only from
them, citing its sources.

## The two pipelines

RAG is really two pipelines that share a vector store:

1. **Ingestion (offline/write path):** load documents -> split into chunks ->
   embed each chunk -> upsert vectors plus metadata into the store.
2. **Query (online/read path):** embed the question -> vector similarity
   search for the top-k chunks -> build a prompt with numbered context
   blocks -> LLM synthesis -> extract citations.

## Chunking

Chunking balances two pressures: chunks must be small enough that retrieval
is precise, yet large enough to carry answerable context. A recursive
character splitter tries separators in priority order (paragraph breaks,
newlines, sentence ends, spaces) so cuts land on natural boundaries. Typical
settings are a chunk size of 500-1000 characters with an overlap of 10-20%
so facts that straddle a boundary survive in at least one chunk.

## Retrieval

The question is embedded with the same model used at ingestion time and
compared against stored vectors, usually by cosine similarity. Top-k of 3-8
chunks is a common sweet spot. Quality upgrades, in rough order of value:
metadata filtering, hybrid search (BM25 + vectors), and a cross-encoder
reranker over the candidate set.

## Grounded synthesis and citations

The prompt presents each retrieved chunk as a numbered block and instructs
the model: answer only from the context, cite blocks inline as [1], [2], and
say "I don't know" when the context is insufficient. Citations are then
parsed from the answer and mapped back to chunk and document IDs, giving
users a verifiable trail and making hallucinations easier to spot.

## Evaluation

RAG quality splits into retrieval and generation metrics. Retrieval is
measured with hit rate@k and MRR against a golden set of question ->
expected-document pairs. Generation is scored for faithfulness and answer
relevance, often with LLM-as-a-Judge rubrics. Evals run in CI against a
frozen corpus, so a chunking or prompt change that hurts quality fails the
build instead of production.

## Common failure modes

Retrieval misses (wrong chunk size, weak embeddings), context stuffing
(top-k too high, model drowns), stale indexes after document updates, and
answers that ignore the context entirely — which is exactly what citation
checks and judge evals are there to catch.
