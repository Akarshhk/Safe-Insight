"""
Text chunking.

Splits extracted document text into overlapping, retrieval-sized pieces while
preserving the metadata needed for citations (filename, page number, chunk
index, character offsets).

Design decisions
----------------
*Why approximate token counts instead of a real tokenizer?*
    Every real tokenizer (tiktoken, HF fast tokenizers) wants to download a
    vocabulary file on first use. That would break the offline guarantee at
    install time and add a dependency for a job that tolerates approximation:
    chunk boundaries only need to be *roughly* the right size. We approximate
    with a words-per-token ratio (``config.WORDS_PER_TOKEN``) and give the LLM a
    generous context margin.

*Why sentence-aware splitting?*
    Cutting mid-sentence produces chunks that read as gibberish in the citation
    panel, and the excerpt shown to the user is the product here. We accumulate
    whole sentences up to the budget, and only hard-split a single sentence when
    it alone exceeds the budget (tables, minified text, OCR runs).

*Why per-page chunking for PDFs?*
    A chunk must map to exactly one page, otherwise the citation cannot name a
    page. ``ingestion.py`` therefore hands us one :class:`TextSegment` per page
    and we never merge across segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app import config

# A sentence ends at . ! ? (optionally followed by quotes/brackets) plus
# whitespace, or at a blank line. Deliberately simple - no NLP dependency.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}")
_WHITESPACE_RUN = re.compile(r"[ \t\r\f\v]+")


@dataclass
class TextSegment:
    """
    One contiguous run of text from a source document, with its own location.

    For a PDF this is a single page; for DOCX/TXT it is the whole document.
    ``page_number`` is 1-based and ``None`` when the format has no pages.
    """

    text: str
    page_number: Optional[int] = None


@dataclass
class Chunk:
    """
    A retrieval unit: the text that gets embedded, plus everything needed to
    cite it back to the user.

    ``chunk_index`` is document-scoped and sequential, which makes chunks stable
    to reference in the audit log even if the FAISS ids change after a rebuild.
    """

    text: str
    chunk_index: int
    page_number: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def approx_tokens(self) -> int:
        """Approximate token count for this chunk (see module docstring)."""
        return estimate_tokens(self.text)


def estimate_tokens(text: str) -> int:
    """
    Estimate the token count of ``text`` without loading a tokenizer.

    Uses whitespace words divided by ``config.WORDS_PER_TOKEN``. Accurate to
    roughly +/-20% on English prose, which is all chunk sizing requires.
    """
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, int(round(words / config.WORDS_PER_TOKEN)))


def normalise_whitespace(text: str) -> str:
    """
    Collapse runs of spaces/tabs and trim trailing space on each line.

    PDF extraction is full of ragged spacing; normalising here means the excerpt
    the user sees in the citation panel is readable, and identical text extracted
    twice produces identical embeddings.
    """
    lines = [_WHITESPACE_RUN.sub(" ", line).strip() for line in text.splitlines()]
    # Preserve paragraph breaks but drop runs of 3+ blank lines.
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentence-ish pieces, dropping empties."""
    pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY.split(text)]
    return [piece for piece in pieces if piece]


def _split_oversized(sentence: str, max_tokens: int) -> List[str]:
    """
    Hard-split a single 'sentence' that is larger than the whole chunk budget.

    Happens with tables, code blocks and OCR output that contains no punctuation.
    We fall back to word-count slicing, which is the only option left.
    """
    words = sentence.split()
    words_per_chunk = max(1, int(max_tokens * config.WORDS_PER_TOKEN))
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def chunk_segment(
    segment: TextSegment,
    start_index: int,
    chunk_size_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk a single :class:`TextSegment`.

    Args:
        segment: the page (or whole document) to chunk.
        start_index: value for the first chunk's ``chunk_index``; the caller
            keeps this running across segments so indices are document-scoped.
        chunk_size_tokens: target size, defaults to ``config.CHUNK_SIZE_TOKENS``.
        overlap_tokens: token overlap between consecutive chunks, defaults to
            ``config.CHUNK_OVERLAP_TOKENS``. Overlap stops an answer-bearing
            sentence from being orphaned exactly on a boundary.

    Returns:
        Chunks in reading order. Empty list if the segment has no usable text.
    """
    size = chunk_size_tokens or config.CHUNK_SIZE_TOKENS
    overlap = overlap_tokens if overlap_tokens is not None else config.CHUNK_OVERLAP_TOKENS
    overlap = max(0, min(overlap, size - 1))  # overlap must be smaller than a chunk

    text = normalise_whitespace(segment.text)
    if not text:
        return []

    sentences: List[str] = []
    for sentence in split_sentences(text):
        if estimate_tokens(sentence) > size:
            sentences.extend(_split_oversized(sentence, size))
        else:
            sentences.append(sentence)

    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_tokens = 0
    index = start_index

    def flush() -> List[str]:
        """Emit the buffered sentences as a chunk; return the overlap tail."""
        nonlocal index
        if not buffer:
            return []
        body = " ".join(buffer)
        # char offsets are relative to the normalised segment text, and are what
        # lets the UI highlight the excerpt inside the full source later.
        char_start = text.find(buffer[0])
        if char_start < 0:
            char_start = 0
        chunks.append(
            Chunk(
                text=body,
                chunk_index=index,
                page_number=segment.page_number,
                char_start=char_start,
                char_end=char_start + len(body),
            )
        )
        index += 1

        # Build the overlap tail: trailing sentences worth ~`overlap` tokens.
        if overlap <= 0:
            return []
        tail: List[str] = []
        tail_tokens = 0
        for sentence in reversed(buffer):
            sentence_tokens = estimate_tokens(sentence)
            if tail_tokens + sentence_tokens > overlap and tail:
                break
            tail.insert(0, sentence)
            tail_tokens += sentence_tokens
        # Never let the tail become the entire chunk, or we would loop forever.
        return tail if len(tail) < len(buffer) else []

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if buffer and buffer_tokens + sentence_tokens > size:
            buffer = flush()
            buffer_tokens = sum(estimate_tokens(s) for s in buffer)
        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        flush()

    return chunks


def chunk_document(
    segments: List[TextSegment],
    chunk_size_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk every segment of a document, keeping ``chunk_index`` running across
    segments so it is unique and ordered within the document.
    """
    all_chunks: List[Chunk] = []
    next_index = 0
    for segment in segments:
        produced = chunk_segment(segment, next_index, chunk_size_tokens, overlap_tokens)
        all_chunks.extend(produced)
        next_index += len(produced)
    return all_chunks


__all__ = [
    "TextSegment",
    "Chunk",
    "estimate_tokens",
    "normalise_whitespace",
    "split_sentences",
    "chunk_segment",
    "chunk_document",
]
