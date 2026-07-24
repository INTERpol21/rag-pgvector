"""Folder connector: scan passes, dedup, resilience and the watch loop."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.db.store import MemoryVectorStore
from app.services.embeddings import HashingEmbedder
from app.services.watcher import _document_id, scan_once, watch_forever

log = logging.getLogger("test.watcher")


def make_deps() -> tuple[MemoryVectorStore, HashingEmbedder]:
    return MemoryVectorStore(), HashingEmbedder(dim=32)


async def run_scan(root: Path, store: MemoryVectorStore, embedder: HashingEmbedder) -> int:
    return await scan_once(
        root, store=store, embedder=embedder, chunk_size=800, chunk_overlap=150, logger=log
    )


async def test_scan_ingests_supported_files_and_subdirs(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# RRF\nReciprocal rank fusion merges rankings.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("pgvector stores embeddings in postgres")
    (tmp_path / "ignored.jpg").write_bytes(b"\x00\x01")  # unsupported extension

    store, embedder = make_deps()
    indexed = await run_scan(tmp_path, store, embedder)

    assert indexed == 2
    stats = await store.stats()
    assert stats["documents"] == 2
    query = (await embedder.embed(["reciprocal rank fusion"]))[0]
    results = await store.search(query, top_k=1)
    assert results[0].document_id == "notes"


async def test_rescan_of_unchanged_files_is_free(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("stable content")
    store, embedder = make_deps()
    assert await run_scan(tmp_path, store, embedder) == 1
    assert await run_scan(tmp_path, store, embedder) == 0  # hash dedup skipped it


async def test_changed_file_is_reindexed_under_the_same_id(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("first version")
    store, embedder = make_deps()
    await run_scan(tmp_path, store, embedder)
    target.write_text("second version, changed")
    assert await run_scan(tmp_path, store, embedder) == 1
    assert (await store.stats())["documents"] == 1  # same id, updated in place


async def test_one_broken_file_does_not_stop_the_rest(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf")
    (tmp_path / "fine.md").write_text("healthy document")
    store, embedder = make_deps()
    indexed = await run_scan(tmp_path, store, embedder)
    assert indexed == 1  # the good file landed, the bad one was logged and skipped


async def test_document_ids_keep_subdirectories_distinct(tmp_path: Path) -> None:
    assert _document_id(tmp_path, tmp_path / "a" / "readme.md") == "a/readme"
    assert _document_id(tmp_path, tmp_path / "b" / "readme.md") == "b/readme"


async def test_watch_loop_survives_scan_crashes_and_cancels_cleanly(tmp_path: Path) -> None:
    store, embedder = make_deps()

    class ExplodingStore(MemoryVectorStore):
        async def content_hashes(self, document_ids):  # type: ignore[override]
            raise RuntimeError("db down")

    (tmp_path / "doc.md").write_text("content")
    task = asyncio.create_task(
        watch_forever(
            tmp_path,
            store=ExplodingStore(),
            embedder=embedder,
            chunk_size=800,
            chunk_overlap=150,
            interval_s=0.01,
            logger=log,
        )
    )
    await asyncio.sleep(0.05)  # a few crashing passes
    assert not task.done()  # the loop survived them
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
