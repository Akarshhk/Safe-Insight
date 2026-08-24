"""
Document ingestion: bytes on disk -> text segments -> chunks.

Supported formats and the library used for each:

======  ==================  ====================================================
Format  Library             Notes
======  ==================  ====================================================
.pdf    PyMuPDF (``fitz``)  One :class:`~app.chunking.TextSegment` per page, so
                            every chunk can name a real page number.
.docx   python-docx         Paragraphs plus table cells, flattened in document
                            order. No page numbers exist in the DOCX model.
.txt    stdlib              Read as UTF-8 with a latin-1 fallback.
.md     stdlib              Treated as plain text; Markdown syntax is left in
                            place because it carries structure the LLM can use.
======  ==================  ====================================================

All three libraries are pure local parsers - none of them opens a socket.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app import config
from app.chunking import Chunk, TextSegment, chunk_document


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is outside ``config.SUPPORTED_EXTENSIONS``."""


class ExtractionError(RuntimeError):
    """Raised when a supported file cannot be parsed (corrupt, encrypted, empty)."""


@dataclass
class IngestedDocument:
    """
    The result of ingesting one file: identity, provenance, and its chunks.

    ``doc_id`` is a UUID rather than the filename so two uploads of different
    files with the same name never collide, and so deleting a document is an
    exact-match operation in the vector store.
    """

    doc_id: str
    filename: str
    stored_path: Path
    file_type: str
    size_bytes: int
    sha256: str
    page_count: Optional[int]
    ingested_at: str
    chunks: List[Chunk]


# --------------------------------------------------------------------------- #
# Format-specific extraction
# --------------------------------------------------------------------------- #
def _extract_pdf(path: Path) -> List[TextSegment]:
    """
    Extract one segment per PDF page using PyMuPDF.

    Pages that yield no text (pure scans) are skipped rather than erroring: a
    mixed document should still ingest its text pages. If *every* page is empty
    the caller raises, which is the signal that OCR would be needed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExtractionError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        ) from exc

    segments: List[TextSegment] = []
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ExtractionError(
                    f"{path.name} is password-protected; decrypt it before ingesting."
                )
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    segments.append(TextSegment(text=text, page_number=page_number))
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        raise ExtractionError(f"Could not read PDF {path.name}: {exc}") from exc
    return segments


def _extract_docx(path: Path) -> List[TextSegment]:
    """
    Extract a DOCX as a single segment.

    DOCX has no page concept (pagination happens at render time), so citations
    for these documents fall back to the chunk index. Table cells are included
    because contracts and compliance documents put the important numbers there.
    """
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExtractionError(
            "python-docx is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not read DOCX {path.name}: {exc}") from exc

    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    body = "\n\n".join(parts)
    return [TextSegment(text=body, page_number=None)] if body.strip() else []


def _extract_text(path: Path) -> List[TextSegment]:
    """Read a plain-text or Markdown file, tolerating non-UTF-8 encodings."""
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Common with Windows-authored .txt files; latin-1 never fails.
        body = path.read_text(encoding="latin-1")
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not read {path.name}: {exc}") from exc
    return [TextSegment(text=body, page_number=None)] if body.strip() else []

def _extract_image(path: Path) -> List[TextSegment]:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ExtractionError(
            "pytesseract and Pillow are not installed. Run: pip install -r requirements.txt"
        ) from exc

    # Quick test to see if tesseract binary is available on the system
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise ExtractionError(
            "Tesseract OCR is not installed or not in PATH. Please install Tesseract (e.g. winget install tesseract-ocr) to support image paste."
        ) from exc

    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as exc:
        raise ExtractionError(f"Could not OCR image {path.name}: {exc}") from exc
        
    return [TextSegment(text=text, page_number=None)] if text.strip() else []


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_text,
    ".md": _extract_text,
    ".png": _extract_image,
    ".jpg": _extract_image,
    ".jpeg": _extract_image,
}


def extract_segments(path: Path) -> List[TextSegment]:
    """
    Dispatch to the right extractor for ``path``'s extension.

    Raises:
        UnsupportedFormatError: extension not in ``config.SUPPORTED_EXTENSIONS``.
        ExtractionError: the file is supported but unreadable or empty.
    """
    suffix = path.suffix.lower()
    if suffix not in _EXTRACTORS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix}'. Supported: "
            f"{', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        )
    segments = _EXTRACTORS[suffix](path)
    if not segments:
        raise ExtractionError(
            f"No extractable text found in {path.name}. If this is a scanned "
            f"document it needs OCR first - Safe Insight does not bundle an OCR "
            f"engine (see README, 'Known limitations')."
        )
    return segments


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_sha256(path: Path) -> str:
    """
    Content hash of a file, streamed so large PDFs do not load into memory.

    Used to detect re-uploads of an identical document so the vector store can
    refuse the duplicate instead of double-weighting it in retrieval.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def store_upload(source_path: Path, original_filename: str, doc_id: str) -> Path:
    """
    Copy an uploaded file into ``config.UPLOADS_DIR`` under its ``doc_id``.

    Keeping the original is what makes a citation *verifiable*: the user can open
    the exact file the excerpt came from. The stored name is
    ``<doc_id><ext>`` so a hostile filename cannot escape the uploads directory.
    """
    config.ensure_directories()
    suffix = Path(original_filename).suffix.lower()
    destination = config.UPLOADS_DIR / f"{doc_id}{suffix}"
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def ingest_file(
    source_path: Path,
    original_filename: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> IngestedDocument:
    """
    Full ingestion of one file: validate -> store -> extract -> chunk.

    Embedding and indexing are deliberately *not* done here; ``main.py`` calls
    :mod:`app.embeddings` and :mod:`app.vector_store` next. Keeping parsing
    separate from indexing means ingestion can be unit-tested with no model
    loaded at all.

    Args:
        source_path: file to ingest (usually a temp file written by the API).
        original_filename: name to show the user; defaults to ``source_path.name``.
        doc_id: reuse an existing id (re-index); defaults to a fresh UUID4.

    Returns:
        An :class:`IngestedDocument` with its chunks populated.
    """
    source_path = Path(source_path)
    filename = original_filename or source_path.name
    suffix = Path(filename).suffix.lower()

    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix}'. Supported: "
            f"{', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        )

    size_bytes = source_path.stat().st_size
    if size_bytes == 0:
        raise ExtractionError(f"{filename} is empty.")
    if size_bytes > config.MAX_UPLOAD_BYTES:
        raise ExtractionError(
            f"{filename} is {size_bytes / 1_048_576:.1f} MB, over the "
            f"{config.MAX_UPLOAD_BYTES / 1_048_576:.0f} MB limit."
        )

    document_id = doc_id or str(uuid.uuid4())
    stored_path = store_upload(source_path, filename, document_id)

    segments = extract_segments(stored_path)
    chunks = chunk_document(segments)
    if not chunks:
        raise ExtractionError(f"{filename} produced no chunks after text extraction.")

    # Stamp each chunk with its document so downstream code never has to guess.
    for chunk in chunks:
        chunk.extra.update({"doc_id": document_id, "filename": filename})

    page_numbers = [s.page_number for s in segments if s.page_number is not None]

    return IngestedDocument(
        doc_id=document_id,
        filename=filename,
        stored_path=stored_path,
        file_type=suffix.lstrip("."),
        size_bytes=size_bytes,
        sha256=compute_sha256(stored_path),
        page_count=max(page_numbers) if page_numbers else None,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunks=chunks,
    )


__all__ = [
    "IngestedDocument",
    "UnsupportedFormatError",
    "ExtractionError",
    "extract_segments",
    "compute_sha256",
    "store_upload",
    "ingest_file",
]
