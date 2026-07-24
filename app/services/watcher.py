"""Folder connector: auto-ingest documents dropped into a watched directory.

Point ``INGEST_WATCH_DIR`` at a folder (in the umbrella stack: a bind-mounted
``./dropbox``) and every supported file (md/txt/pdf/docx) in it lands in the
knowledge base within one poll interval — no API call, no UI. Content-hash
dedup makes rescans free: an unchanged file is skipped before any chunking or
embedding work, so polling the whole tree every few seconds costs one hash
comparison per file.

Polling, not inotify: filesystem events are unreliable across Docker
bind mounts (macOS/Windows hosts especially), and at demo scale a periodic
walk is simpler and equally fast.

Deliberate limits: deletions are NOT propagated (the VectorStore has no
delete surface; removing a file leaves its chunks searchable), and a file
that fails to parse is logged and skipped — one bad PDF must not stop the
rest of the drop folder from ingesting.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.db.store import VectorStore
from app.schemas import MAX_TEXT_CHARS, DocumentIn
from app.services.embeddings import Embedder
from app.services.extract import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, extract_text
from app.services.ingest import ingest_documents


def _document_id(root: Path, path: Path) -> str:
    """Stable id from the file's location: ``docs/notes.md`` -> ``docs/notes``.

    The relative path (not just the stem) keeps ``a/readme.md`` and
    ``b/readme.md`` distinct; the suffix is dropped so replacing ``notes.txt``
    with ``notes.md`` updates the same document instead of duplicating it.
    """
    return path.relative_to(root).with_suffix("").as_posix()


async def scan_once(
    root: Path,
    *,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    logger: logging.Logger,
) -> int:
    """One pass over the tree; returns how many documents were (re)indexed."""
    indexed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                logger.warning("watcher: %s is over the %d-byte cap, skipped",
                               path.name, MAX_UPLOAD_BYTES)
                continue
            text = extract_text(path.name, path.read_bytes())
            if not text.strip():
                continue
            document = DocumentIn(
                id=_document_id(root, path),
                title=path.stem,
                text=text[:MAX_TEXT_CHARS],
                metadata={"connector": "folder", "filename": path.name},
                source="local",
            )
            result = await ingest_documents(
                [document],
                store=store,
                embedder=embedder,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if result.chunks_indexed:
                indexed += 1
                logger.info("watcher: indexed %s (%d chunks)",
                            document.id, result.chunks_indexed)
        except Exception:  # noqa: BLE001 - one bad file must not stop the folder
            logger.warning("watcher: failed to ingest %s", path.name, exc_info=True)
    return indexed


async def watch_forever(
    root: Path,
    *,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    interval_s: float,
    logger: logging.Logger,
) -> None:
    """Poll the folder until cancelled; scan crashes are logged, never fatal."""
    logger.info("watcher: watching %s every %.0fs", root, interval_s)
    while True:
        try:
            await scan_once(
                root,
                store=store,
                embedder=embedder,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                logger=logger,
            )
        except Exception:  # noqa: BLE001 - the loop itself must survive anything
            logger.warning("watcher: scan pass failed", exc_info=True)
        await asyncio.sleep(interval_s)
