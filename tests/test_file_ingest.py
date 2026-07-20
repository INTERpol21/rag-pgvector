"""File upload ingest (md/txt/pdf/docx): extraction + the /ingest/file endpoint."""

from __future__ import annotations

import io

import docx
import pytest

from app.services.extract import ExtractionError, UnsupportedFileError, extract_text


def _make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a single text run pypdf can extract."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = (f"BT /F1 18 Tf 20 250 Td ({text}) Tj ET").encode()
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return pdf


def _make_docx(*paragraphs: str) -> bytes:
    document = docx.Document()
    for para in paragraphs:
        document.add_paragraph(para)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


# -- extraction unit tests ----------------------------------------------------


def test_extract_txt_and_md_decode_utf8() -> None:
    assert extract_text("a.txt", b"hello world") == "hello world"
    assert extract_text("b.md", "# Ttl\n\nтело".encode()) == "# Ttl\n\nтело"


def test_extract_docx_joins_paragraphs() -> None:
    text = extract_text("c.docx", _make_docx("line one", "", "line two"))
    assert "line one" in text
    assert "line two" in text


def test_extract_pdf_reads_text() -> None:
    assert "Hello PDF ingest" in extract_text("d.pdf", _make_pdf("Hello PDF ingest"))


def test_extract_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileError):
        extract_text("x.exe", b"MZ\x00\x00")


def test_extract_corrupt_pdf_raises_extraction_error() -> None:
    # Catches a corrupt upload crashing the request instead of a clean 422.
    with pytest.raises(ExtractionError):
        extract_text("bad.pdf", b"not a real pdf at all")


# -- /ingest/file endpoint tests ---------------------------------------------

QUESTION = "pgvector cosine distance operator vector_cosine_ops"


async def test_ingest_file_txt_is_indexed_and_local(client):
    body = (
        await client.post(
            "/ingest/file",
            files={"file": ("notes.txt", QUESTION.encode(), "text/plain")},
            data={"source": "local"},
        )
    ).json()
    assert body["chunks_indexed"] >= 1

    result = (await client.post("/query", json={"question": QUESTION, "top_k": 4})).json()
    assert result["retrieved"], result
    assert result["retrieved"][0]["source"] == "local"


async def test_ingest_file_docx_is_indexed(client):
    resp = await client.post(
        "/ingest/file",
        files={
            "file": (
                "notes.docx",
                _make_docx(QUESTION),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1


async def test_ingest_file_unsupported_type_returns_415(client):
    resp = await client.post(
        "/ingest/file",
        files={"file": ("evil.exe", b"MZ\x00binary", "application/octet-stream")},
    )
    assert resp.status_code == 415


async def test_ingest_file_empty_text_returns_422(client):
    resp = await client.post(
        "/ingest/file",
        files={"file": ("blank.txt", b"   \n  \t ", "text/plain")},
    )
    assert resp.status_code == 422


async def test_ingest_file_oversized_text_returns_413_not_500(client):
    """Extracted text over the per-document limit is a clean 413, not a 500.

    Regression: constructing DocumentIn in the route with over-limit text raised
    pydantic ValidationError as an unhandled 500 instead of rejecting cleanly.
    """
    big = ("word " * 250_000).encode()  # ~1.25 MB > MAX_TEXT_CHARS
    resp = await client.post(
        "/ingest/file", files={"file": ("big.txt", big, "text/plain")}
    )
    assert resp.status_code == 413


async def test_ingest_file_overlong_filename_title_returns_422_not_500(client):
    """A pathologically long filename (title > limit) is a 422, not a 500."""
    resp = await client.post(
        "/ingest/file",
        files={"file": ("x" * 2000 + ".txt", b"real content here", "text/plain")},
    )
    assert resp.status_code == 422
