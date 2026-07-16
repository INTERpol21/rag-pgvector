import httpx
import pytest

from app.embeddings import HashingEmbedder
from app.llm import MockLLM
from app.main import create_app
from app.settings import Settings
from app.store import MemoryVectorStore


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def app(store, embedder):
    settings = Settings(
        embeddings_backend="hash", store_backend="memory", llm_backend="mock"
    )
    return create_app(settings, store=store, embedder=embedder, llm=MockLLM())


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


DOCS = [
    {
        "id": "gateway",
        "title": "Gateway Guide",
        "text": (
            "An LLM gateway centralizes authentication, rate limiting and cost "
            "tracking. A token bucket limiter refills at a fixed rate and "
            "rejects requests with HTTP 429 when the bucket is empty. "
            "Streaming uses Server-Sent Events."
        ),
    },
    {
        "id": "pgvector",
        "title": "pgvector Guide",
        "text": (
            "pgvector adds a vector column type to Postgres. Cosine distance "
            "uses the <=> operator with the vector_cosine_ops opclass. IVFFlat "
            "and HNSW indexes provide approximate nearest neighbor search; "
            "tune lists and probes for recall."
        ),
    },
    {
        "id": "espresso",
        "title": "Espresso Handbook",
        "text": (
            "Espresso extraction depends on grind size, dose and water "
            "temperature. A typical shot uses eighteen grams of coffee, a "
            "twenty five second pour and ninety three degree water. Crema "
            "signals fresh beans."
        ),
    },
]


@pytest.fixture
async def ingested_client(client):
    resp = await client.post("/ingest", json={"documents": DOCS})
    assert resp.status_code == 200, resp.text
    return client
