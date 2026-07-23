"""Application settings loaded from environment variables / .env."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every field maps to an UPPER_CASE environment variable of the same name
    (e.g. ``store_backend`` <- ``STORE_BACKEND``). Defaults are chosen so the
    service runs fully offline: in-memory vector store, hashing embedder and
    a deterministic mock LLM.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API ---
    # Bind all interfaces: intended for the containerized service (Docker/compose).
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8081
    # Comma-separated bearer tokens accepted on /ingest, /query and /stats.
    rag_api_keys: str = "demo-key"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Embeddings ---
    embeddings_backend: str = "hash"  # hash | semantic | openai
    embedding_dim: int = 256  # used by the hashing embedder
    embedding_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # --- Vector store ---
    store_backend: str = "memory"  # memory | pgvector
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # --- Retrieval ---
    # hybrid = RRF merge of vector search and keyword search (BM25 in memory,
    # Postgres FTS in pgvector); vector = cosine-only, the pre-hybrid behavior.
    search_mode: Literal["vector", "hybrid"] = "hybrid"
    reranker: str = "none"  # none | mock | llm

    # --- LLM synthesis ---
    llm_backend: str = "mock"  # mock | grounded | openai
    llm_base_url: str = "http://localhost:8080/v1"  # sibling llm-gateway
    llm_api_key: str = "demo-key"
    llm_model: str = "mock-small"
    llm_timeout_s: float = 30.0

    # --- Evals ---
    judge_backend: str = "mock"  # mock | openai (used by evals/run_evals.py)
