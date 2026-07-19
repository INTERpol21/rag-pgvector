#!/usr/bin/env python3
"""Offline eval harness for the RAG pipeline.

Builds a fresh in-memory index from ``data/``, runs the full
chunk -> embed -> retrieve -> synthesize -> cite pipeline for every item in
``evals/golden.jsonl`` and reports:

* **hit_rate@k** — was the expected document among the retrieved chunks;
* **citation_presence** — did the answer carry at least one resolvable [n];
* **judge_score** — LLM-as-a-Judge answer quality on a 1-5 scale
  (deterministic ``MockJudge`` by default; set ``JUDGE_BACKEND=openai`` to
  judge with a real model through the llm-gateway).

Usage::

    python evals/run_evals.py [--limit N] [--top-k K] [--min-hit-rate 0.7]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python evals/run_evals.py` from repo root
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_text  # noqa: E402
from app.citations import extract_citations  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.errors import ProviderError  # noqa: E402
from app.llm import LLM, MockLLM  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.store import (  # noqa: E402
    ChunkRecord,
    DocumentRecord,
    MemoryVectorStore,
    search_with_mode,
)

GOLDEN_PATH = ROOT / "evals" / "golden.jsonl"
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "evals" / "report.md"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


class Judge(Protocol):
    name: str

    async def score(self, question: str, reference: str, answer: str) -> int:
        """Return an integer quality score in [1, 5]."""
        ...


class MockJudge:
    """Deterministic stand-in for an LLM judge.

    Scores by content-word overlap between the generated answer and the
    reference, mapped monotonically onto 1-5. It cannot assess reasoning or
    faithfulness; it exists so evals run offline and deterministically in
    CI. Set ``JUDGE_BACKEND=openai`` to judge with a real model through the
    gateway.
    """

    name = "mock"

    async def score(self, question: str, reference: str, answer: str) -> int:
        ref_tokens = _content_tokens(reference)
        if not ref_tokens:
            return 1
        overlap = len(ref_tokens & _content_tokens(answer)) / len(ref_tokens)
        return 1 + min(4, int(overlap * 5))


JUDGE_PROMPT = (
    "You are grading a RAG system. Given a question, a reference answer and "
    "a candidate answer, rate the candidate's correctness and groundedness "
    "on a 1-5 scale (5 = fully correct and grounded, 1 = wrong or "
    "unsupported). Reply with ONLY the integer.\n\n"
    "Question: {question}\n\nReference answer: {reference}\n\n"
    "Candidate answer: {answer}\n\nScore:"
)


class OpenAIJudge:
    """LLM-as-a-Judge via an OpenAI-compatible chat endpoint."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def score(self, question: str, reference: str, answer: str) -> int:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, reference=reference, answer=answer
                    ),
                }
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise ProviderError(f"judge call failed: {exc}") from exc
        match = re.search(r"[1-5]", content)
        return int(match.group()) if match else 1


def build_judge(settings: Settings) -> Judge:
    backend = settings.judge_backend.lower()
    if backend == "mock":
        return MockJudge()
    if backend == "openai":
        return OpenAIJudge(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"unknown JUDGE_BACKEND: {backend!r}")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass
class ItemResult:
    question: str
    expected_document_id: str
    retrieved_document_ids: list[str]
    hit: bool
    citations: int
    judge_score: int
    answer: str


@dataclass
class EvalSummary:
    top_k: int
    documents: int
    chunks: int
    judge_name: str
    items: list[ItemResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return sum(i.hit for i in self.items) / len(self.items) if self.items else 0.0

    @property
    def citation_presence(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.citations > 0 for i in self.items) / len(self.items)

    @property
    def avg_judge_score(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.judge_score for i in self.items) / len(self.items)


_REQUIRED_GOLDEN_KEYS = ("question", "expected_document_id", "reference_answer")


def load_golden(path: Path, limit: int | None = None) -> list[dict]:
    items: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"golden line {lineno} is not valid JSON: {exc}") from exc
        missing = [k for k in _REQUIRED_GOLDEN_KEYS if k not in row]
        if missing:
            raise ValueError(
                f"golden line {lineno} is missing required key(s): {', '.join(missing)}"
            )
        items.append(row)
    return items[:limit] if limit is not None else items


async def ingest_corpus(
    store: MemoryVectorStore,
    embedder: HashingEmbedder,
    data_dir: Path,
    settings: Settings,
) -> tuple[int, int]:
    """Index every ``*.md`` in ``data_dir``; document id = file stem."""
    n_docs = n_chunks = 0
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first_line = text.strip().splitlines()[0] if text.strip() else path.stem
        title = first_line.lstrip("# ").strip() or path.stem
        pieces = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        vectors = await embedder.embed(pieces)
        document = DocumentRecord(id=path.stem, title=title, metadata={"source": path.name})
        chunks = [
            ChunkRecord(
                id=f"{path.stem}:{i}",
                document_id=path.stem,
                ord=i,
                content=piece,
                embedding=vec,
            )
            for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=False))
        ]
        await store.upsert(document, chunks)
        n_docs += 1
        n_chunks += len(chunks)
    return n_docs, n_chunks


async def run_evals(
    golden_path: Path = GOLDEN_PATH,
    data_dir: Path = DATA_DIR,
    top_k: int = 4,
    limit: int | None = None,
    llm: LLM | None = None,
    judge: Judge | None = None,
) -> EvalSummary:
    """Run the full pipeline on the golden set with a fresh in-memory index."""
    settings = Settings()
    embedder = HashingEmbedder(dim=settings.embedding_dim)
    store = MemoryVectorStore()
    llm = llm or MockLLM()
    judge = judge or build_judge(settings)

    n_docs, n_chunks = await ingest_corpus(store, embedder, data_dir, settings)
    summary = EvalSummary(
        top_k=top_k, documents=n_docs, chunks=n_chunks, judge_name=judge.name
    )

    for item in load_golden(golden_path, limit):
        question = item["question"]
        query_vec = (await embedder.embed([question]))[0]
        # Same dispatch as the API: SEARCH_MODE decides vector vs hybrid.
        retrieved = await search_with_mode(
            store, settings.search_mode, query_vec, question, top_k
        )
        result = await llm.answer(question, retrieved)
        citations = extract_citations(result.answer, retrieved)
        doc_ids = [c.document_id for c in retrieved]
        summary.items.append(
            ItemResult(
                question=question,
                expected_document_id=item["expected_document_id"],
                retrieved_document_ids=doc_ids,
                hit=item["expected_document_id"] in doc_ids,
                citations=len(citations),
                judge_score=await judge.score(
                    question, item["reference_answer"], result.answer
                ),
                answer=result.answer,
            )
        )
    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def metrics_table(summary: EvalSummary) -> str:
    rows = [
        ("items", f"{len(summary.items)}"),
        (f"hit_rate@{summary.top_k}", f"{summary.hit_rate:.2f}"),
        ("citation_presence", f"{summary.citation_presence:.2f}"),
        (f"judge_score (1-5, {summary.judge_name})", f"{summary.avg_judge_score:.2f}"),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name.ljust(width)}  {value}" for name, value in rows)


def render_report(summary: EvalSummary) -> str:
    lines = [
        "# Eval report",
        "",
        f"Corpus: {summary.documents} documents / {summary.chunks} chunks "
        f"(in-memory index, hashing embedder). Judge: `{summary.judge_name}`.",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| hit_rate@{summary.top_k} | {summary.hit_rate:.2f} |",
        f"| citation_presence | {summary.citation_presence:.2f} |",
        f"| judge_score (1-5) | {summary.avg_judge_score:.2f} |",
        "",
        "## Per-item results",
        "",
        "| question | expected doc | hit | citations | judge |",
        "|---|---|---|---|---|",
    ]
    for item in summary.items:
        lines.append(
            f"| {item.question} | {item.expected_document_id} | "
            f"{'yes' if item.hit else 'NO'} | {item.citations} | {item.judge_score} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.0,
        help="exit non-zero if hit_rate falls below this (for CI gates)",
    )
    args = parser.parse_args(argv)

    summary = asyncio.run(
        run_evals(
            golden_path=args.golden, data_dir=args.data, top_k=args.top_k, limit=args.limit
        )
    )

    print(metrics_table(summary))
    args.report.write_text(render_report(summary), encoding="utf-8")
    print(f"\nreport written to {args.report}")

    if summary.hit_rate < args.min_hit_rate:
        print(
            f"FAIL: hit_rate {summary.hit_rate:.2f} < required {args.min_hit_rate:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
