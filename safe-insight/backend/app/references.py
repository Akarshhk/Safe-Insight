"""
Which documents a question is about, when it names them or points at one.

Meaning-based search cannot answer "what is this doc about?": the question has
no content to match, so every passage scores about the same (0.4-0.5 with
bge-small, measured) and an unrelated document can win. These plain-text rules
cover the two ways people refer to a file instead of its content:

* by name - "what's in akarsh_kalapurmut_dsatm.pdf?", "explain the perceptron
  convergence notes" (the filename, or two of its words), and
* by pointing - "this doc", "the attached file", "my resume", "summarize it",
  which mean the document uploaded most recently.

Pure string handling, so it is unit-tested without any model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # avoid importing numpy/faiss just to type-annotate.
    from app.vector_store import DocumentRecord

#: "this doc", "the attached file", "my resume", "summarize ...".
_POINTS_AT_A_DOCUMENT = re.compile(
    r"\b(?:this|that|the|my|attached|uploaded)\s+"
    r"(?:docs?|documents?|files?|pdfs?|attachments?|papers?|reports?|slides?|decks?|resumes?|cvs?)\b"
    r"|\b(?:attached|attachment|uploaded)\b"
    r"|\bsummari[sz]e\b",
    re.IGNORECASE,
)

#: Filename words too common to identify a document on their own.
_WEAK_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "do", "for", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "the", "to", "up", "us", "we",
    "copy", "doc", "docx", "file", "final", "md", "new", "pdf", "ppt", "pptx", "txt",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def names_document(question: str, filename: str) -> bool:
    """
    True when the question contains the filename, or enough of its words.

    "Enough" is two distinctive words ("akarsh kalapurmut", "bi module"), or the
    only one when the name has just one ("resume.pdf" and "my resume").
    """
    question_words = _words(question)
    stem = _words(Path(filename).stem)
    stem_text = " ".join(stem)
    if len(stem_text) >= 4 and f" {stem_text} " in f" {' '.join(question_words)} ":
        return True
    distinctive = {w for w in stem if w not in _WEAK_WORDS and not w.isdigit()}
    if not distinctive:
        return False
    return len(distinctive & set(question_words)) >= min(2, len(distinctive))


def referred_documents(
    question: str, documents: Sequence["DocumentRecord"]
) -> List["DocumentRecord"]:
    """
    The documents a question is about by name or by pointing; empty otherwise.

    Named documents win. "This doc"-style questions get the newest upload,
    because that is almost always the one the user just attached.
    """
    named = [doc for doc in documents if names_document(question, doc.filename)]
    if named:
        return named
    if documents and _POINTS_AT_A_DOCUMENT.search(question):
        return [max(documents, key=lambda doc: doc.ingested_at)]
    return []


__all__ = ["names_document", "referred_documents"]
