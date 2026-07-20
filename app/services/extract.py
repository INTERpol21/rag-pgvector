"""Extract plain text from uploaded documents (md/txt/pdf/docx).

Lightweight, self-hosted, offline parsers: markdown/text decode directly, PDF via
pypdf, DOCX via python-docx. (Docling — the richer 2026 parser — is a planned
upgrade; these cover the common formats without its weight.) Only text is
extracted; layout/images are ignored.
"""

from __future__ import annotations

import io
from pathlib import Path

# Extensions handled by extract_text (lower-case, incl. the dot).
SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".text", ".pdf", ".docx"})
_TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".text"})

# Upload size cap (10 MB): bounds a single request's memory/parse work before the
# extracted text hits the (smaller) per-document text limit.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class UnsupportedFileError(ValueError):
    """Raised when a file's extension is not one we can extract text from."""


class ExtractionError(ValueError):
    """Raised when a supported file is corrupt or unreadable."""


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of ``data``, dispatching on ``filename``'s extension."""
    ext = Path(filename).suffix.lower()
    if ext in _TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace").strip()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    raise UnsupportedFileError(
        f"unsupported file type {ext or '(none)'!r}; supported: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, OSError, ValueError, KeyError) as exc:
        raise ExtractionError(f"could not read DOCX: {exc}") from exc
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
