"""
Citation trail - the project's key differentiator.

A retrieved chunk is not a citation. A citation is a *claim about provenance*
that a compliance officer can check: which file, which page, which words, and
how confident the retrieval was. This module turns
:class:`~app.vector_store.SearchResult` objects into that, and links the
bracketed markers the model writes (``[2]``) back to the source.

Guarantees this module provides
-------------------------------
1. **Stable numbering.** Citation ``index`` is 1-based and matches the numbering
   in the prompt built by :func:`app.llm.build_prompt`, so ``[2]`` in the answer
   is always citation 2 in the response payload.
2. **Verifiable location.** Every citation carries ``filename`` plus either a
   page number (PDF) or a chunk position (DOCX/TXT), and the ``vector_id`` needed
   to fetch the untruncated chunk from ``GET /chunks/{vector_id}``.
3. **Honest excerpts.** Excerpts are truncated on a word boundary and flagged
   with ``truncated``; the full text is always one API call away.
4. **Audited usage.** :func:`extract_cited_indices` reports which citations the
   model actually referenced, so the audit log records both what was retrieved
   and what was used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from app import config

if TYPE_CHECKING:  # import only for type checking - keeps this module free of
    from app.vector_store import SearchResult  # numpy/faiss at import time.

#: Matches inline markers the model writes: [1], [2, 3] or [1][4].
_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    """
    One entry in the citation trail, serialised straight to the frontend.

    Attributes:
        index: 1-based position; matches the ``[n]`` marker in the answer.
        vector_id: primary key of the chunk in the vector store. The UI uses it
            for ``GET /chunks/{vector_id}`` when the user clicks the citation.
        doc_id: owning document, for "open the source file".
        filename: display name of the source document.
        page_number: 1-based page for PDFs, ``None`` for formats without pages.
        chunk_index: 0-based position of the chunk within its document; the
            fallback locator when there is no page number.
        excerpt: possibly-truncated chunk text, ready to display.
        truncated: True when ``excerpt`` is shorter than the stored chunk.
        score: cosine similarity of the chunk to the query, 0-1.
        location: pre-formatted human label, e.g. "contract.pdf, page 4".
        referenced: True when the answer text actually contains this marker.
    """

    index: int
    vector_id: int
    doc_id: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    excerpt: str
    truncated: bool
    score: float
    location: str
    referenced: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict for JSON responses and for the audit log."""
        return asdict(self)


def format_location(filename: str, page_number: Optional[int], chunk_index: int) -> str:
    """
    Human-readable source label.

    PDFs get a page number because that is what a reader can physically check.
    Formats with no pagination fall back to a 1-based chunk position, which is at
    least monotonic through the document.
    """
    if page_number is not None:
        return f"{filename}, page {page_number}"
    return f"{filename}, section {chunk_index + 1}"


def truncate_excerpt(text: str, max_chars: Optional[int] = None) -> Tuple[str, bool]:
    """
    Trim ``text`` for display without cutting a word in half.

    Returns:
        ``(excerpt, was_truncated)``. When truncated, an ellipsis is appended so
        the user can see that more text exists behind the citation click.
    """
    limit = config.CITATION_EXCERPT_CHARS if max_chars is None else max_chars
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean, False
    cut = clean[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:      # only back up to a word boundary if it is near
        cut = cut[:last_space]
    return cut.rstrip(" ,;:.") + " ...", True


def extract_cited_indices(answer: str) -> List[int]:
    """
    Pull the citation numbers the model referenced out of its answer.

    Handles ``[1]``, ``[1, 2]`` and ``[1][2]``. Returns them sorted and
    de-duplicated. Numbers outside the citation range are *not* filtered here -
    :func:`mark_referenced` does that, so a hallucinated ``[9]`` simply marks
    nothing rather than silently disappearing.
    """
    found: set[int] = set()
    for match in _CITATION_MARKER.finditer(answer):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return sorted(found)


def build_citations(
    results: "Sequence[SearchResult]",
    max_excerpt_chars: Optional[int] = None,
) -> List[Citation]:
    """
    Convert retrieval results into the citation trail.

    Order is preserved exactly: ``results[0]`` becomes citation 1, which is the
    contract :func:`app.llm.build_prompt` relies on.
    """
    citations: List[Citation] = []
    for position, result in enumerate(results, start=1):
        chunk = result.chunk
        excerpt, truncated = truncate_excerpt(chunk.text, max_excerpt_chars)
        citations.append(
            Citation(
                index=position,
                vector_id=chunk.vector_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                excerpt=excerpt,
                truncated=truncated,
                score=round(result.score, 4),
                location=format_location(chunk.filename, chunk.page_number, chunk.chunk_index),
            )
        )
    return citations


def mark_referenced(citations: Sequence[Citation], answer: str) -> List[int]:
    """
    Flag which citations the answer actually cites; return those indices.

    Useful two ways: the UI can dim unreferenced sources, and the audit log can
    record the difference between *retrieved* and *relied upon* - a distinction
    an auditor will care about.
    """
    cited = set(extract_cited_indices(answer))
    valid: List[int] = []
    for citation in citations:
        citation.referenced = citation.index in cited
        if citation.referenced:
            valid.append(citation.index)
    return valid


def citation_summary(citations: Sequence[Citation]) -> Dict[str, Any]:
    """Compact provenance summary, stored on the audit record."""
    return {
        "count": len(citations),
        "documents": sorted({c.filename for c in citations}),
        "vector_ids": [c.vector_id for c in citations],
        "referenced_indices": [c.index for c in citations if c.referenced],
        "top_score": max((c.score for c in citations), default=0.0),
    }


__all__ = [
    "Citation",
    "format_location",
    "truncate_excerpt",
    "extract_cited_indices",
    "build_citations",
    "mark_referenced",
    "citation_summary",
]
