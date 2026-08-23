"""
Pydantic request/response models for the HTTP API.

Kept in one place so the shapes here and the TypeScript interfaces in
``frontend/src/types.ts`` can be diffed side by side. If you change a model here,
change its twin there.

(This file is an addition to the structure in the original brief - the brief
listed the pipeline modules only. Splitting the wire format out of ``main.py``
keeps the route handlers readable, which matters because the next agent to touch
this project reads ``main.py`` first.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app import config


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class DocumentOut(BaseModel):
    """One indexed document, as shown in the document management panel."""

    doc_id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    page_count: Optional[int] = None
    chunk_count: int
    ingested_at: str


class DocumentListOut(BaseModel):
    """Response for ``GET /documents``."""

    documents: List[DocumentOut]
    total_chunks: int


class IngestOut(BaseModel):
    """Response for ``POST /documents`` (a successful upload)."""

    doc_id: str
    filename: str
    chunk_count: int
    page_count: Optional[int] = None
    duplicate_of: Optional[str] = Field(
        default=None,
        description="Set when an identical file (same SHA-256) was already indexed.",
    )
    elapsed_ms: int


class DeleteOut(BaseModel):
    """Response for ``DELETE /documents/{doc_id}``."""

    doc_id: str
    removed_vectors: int


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class QueryIn(BaseModel):
    """Request body for ``POST /query``."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1,
        le=config.MAX_TOP_K,
        description="How many chunks to retrieve before generation.",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the cosine-similarity floor for this query.",
    )
    strict_mode: bool = Field(
        default=False,
        description="If true, refuse to answer ungrounded queries. If false, fallback to general knowledge.",
    )


class CitationOut(BaseModel):
    """One citation. Mirrors :class:`app.citations.Citation`."""

    index: int
    vector_id: int
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    excerpt: str
    truncated: bool
    score: float
    location: str
    referenced: bool = False


class QueryOut(BaseModel):
    """Response for ``POST /query`` - the answer plus its full provenance."""

    answer: str
    citations: List[CitationOut]
    #: False when retrieval found nothing above the similarity floor, i.e. the
    #: answer is the standard refusal rather than a grounded response.
    answered_from_context: bool
    grounded: bool
    #: True when no GGUF model is loaded and the answer is the retrieval-only stub.
    stub_mode: bool
    model: str
    embedding_model: Optional[str] = None
    top_k: int
    retrieved_count: int
    latency_ms: int
    audit_id: int


class ChunkOut(BaseModel):
    """Response for ``GET /chunks/{vector_id}`` - the untruncated source text."""

    vector_id: int
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    text: str
    location: str


# --------------------------------------------------------------------------- #
# Health / audit
# --------------------------------------------------------------------------- #
class OfflineCheckOut(BaseModel):
    """Response for ``GET /health/offline-check`` - drives the offline badge."""

    offline_verified: bool
    guard_installed: bool
    strict_mode: bool
    outbound_probe_blocked: Optional[bool] = None
    self_check_detail: Optional[str] = None
    violation_count: int
    recent_violations: List[Dict[str, Any]] = Field(default_factory=list)
    http_client_modules_loaded: List[str] = Field(default_factory=list)
    download_in_progress: bool = False


class HealthOut(BaseModel):
    """Response for ``GET /health``."""

    status: str
    version: str
    offline: OfflineCheckOut
    index: Dict[str, Any]
    llm: Dict[str, Any]
    audit: Dict[str, Any]
    embedding_model_loaded: bool


class AuditEntryOut(BaseModel):
    """One row of the query audit log."""

    id: int
    timestamp_utc: str
    query: str
    answer: str
    top_k: int
    retrieved_ids: List[int]
    citations: List[Dict[str, Any]]
    documents_touched: List[str]
    model: str
    embedding_model: Optional[str] = None
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stub_mode: bool = False
    grounded: bool = True
    record_hash: str


class AuditListOut(BaseModel):
    """Response for ``GET /audit/queries``."""

    entries: List[AuditEntryOut]
    total: int


class AuditVerifyOut(BaseModel):
    """Response for ``GET /audit/verify`` - hash-chain integrity result."""

    valid: bool
    checked: int
    total: int
    detail: str
    broken_at_id: Optional[int] = None


class ErrorOut(BaseModel):
    """Uniform error envelope so the frontend has one shape to handle."""

    detail: str
    code: Optional[str] = None

DocumentOut.model_rebuild()
DocumentListOut.model_rebuild()
IngestOut.model_rebuild()
DeleteOut.model_rebuild()
QueryIn.model_rebuild()
CitationOut.model_rebuild()
QueryOut.model_rebuild()
ChunkOut.model_rebuild()
OfflineCheckOut.model_rebuild()
HealthOut.model_rebuild()
AuditEntryOut.model_rebuild()
AuditListOut.model_rebuild()
AuditVerifyOut.model_rebuild()
ErrorOut.model_rebuild()
