"""
Safe Insight backend - FastAPI entrypoint.

Run directly (this is also exactly what the Tauri shell spawns)::

    python -m app.main

or with uvicorn::

    uvicorn app.main:app --host 127.0.0.1 --port 8765

Startup order matters and is not accidental:

1. :mod:`app.config` is imported first (sets the HuggingFace offline env vars).
2. :func:`app.network_guard.install_guard` patches the socket API **before** any
   ML library is imported, so nothing can open a remote socket even once.
3. The FAISS index and audit DB are opened; the embedding model and LLM are
   warmed up so the first question is not the slow one.

API surface
-----------
========================================  ==========================================
``GET  /health``                          full status blob
``GET  /health/offline-check``            offline badge data
``POST /documents``                       upload + ingest + index one file
``GET  /documents``                       list indexed documents
``DELETE /documents/{doc_id}``            remove a document and its vectors
``GET  /documents/{doc_id}/chunks``       every chunk of one document
``POST /query``                           ask a question, get answer + citations
``GET  /chunks/{vector_id}``              untruncated source text for a citation
``GET  /audit/queries``                   recent audit entries
``GET  /audit/verify``                    verify the audit hash chain
``POST /audit/export``                    dump the audit log to JSON Lines
========================================  ==========================================
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List

# 1. config first - it sets the offline environment variables on import.
from app import config

# 2. network guard second - patch sockets before anything else can use them.
from app import network_guard

network_guard.install_guard()

# 3. Everything else may now be imported safely.
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import audit_log, citations as citations_mod, embeddings, ingestion, llm, projects_db, setup, vector_store
from app.schemas import (
    AuditListOut,
    AuditVerifyOut,
    ChunkOut,
    DeleteOut,
    DocumentListOut,
    DocumentOut,
    HealthOut,
    IngestOut,
    OfflineCheckOut,
    QueryIn,
    QueryOut,
)
from app.vector_store import VectorStoreError

VERSION = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("safe_insight")


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up and shut-down work for the whole backend."""
    logger.info("Safe Insight backend %s starting", VERSION)
    config.ensure_directories()

    # Prove the kill-switch works before we claim to be offline.
    network_guard.run_self_check()
    logger.info(network_guard.describe())

    audit_log.init_db()
    projects_db.init_db()

    # Ensure all existing projects have a directory so older projects don't fail
    for proj in projects_db.list_projects():
        (config.PROJECTS_DIR / proj["id"]).mkdir(parents=True, exist_ok=True)

    # Warm the embedding model first: the vector store needs its dimension.
    embeddings.warm_up()

    llm.warm_up()
    logger.info("Ready on http://%s:%d", config.HOST, config.PORT)

    yield

    logger.info("Safe Insight backend shutting down")


app = FastAPI(
    title="Safe Insight",
    version=VERSION,
    description="Offline-only RAG backend. No outbound network access.",
    lifespan=lifespan,
)

# Loopback bind plus this origin allowlist. The backend is not reachable from
# another machine even if the firewall is wide open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(setup.router)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut, tags=["health"])
def health() -> Dict[str, Any]:
    """Everything the UI needs to render its status bar in one round trip."""
    return {
        "status": "ok",
        "version": VERSION,
        "offline": network_guard.get_status(),
        "index": {"loaded": True}, # legacy payload
        "llm": llm.status(),
        "audit": audit_log.stats(),
        "embedding_model_loaded": embeddings.is_loaded(),
    }


@app.get("/health/offline-check", response_model=OfflineCheckOut, tags=["health"])
def offline_check(
    rerun: bool = Query(
        default=False,
        description="Re-run the live outbound probe instead of returning the cached result.",
    )
) -> Dict[str, Any]:
    """
    Offline verification, polled by the "Offline Mode: Verified" badge.

    Pass ``?rerun=true`` for a live probe - that is the button to press during a
    demo, right after switching the machine to airplane mode.
    """
    return network_guard.get_status(rerun_self_check=rerun)


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
from pydantic import BaseModel

class CreateProjectIn(BaseModel):
    name: str

@app.post("/projects", tags=["projects"], status_code=201)
def create_project(payload: CreateProjectIn) -> Dict[str, Any]:
    return projects_db.create_project(payload.name)

@app.get("/projects", tags=["projects"])
def list_projects() -> List[Dict[str, Any]]:
    return projects_db.list_projects()

@app.get("/projects/{project_id}", tags=["projects"])
def get_project(project_id: str) -> Dict[str, Any]:
    proj = projects_db.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    store = vector_store.get_store(project_id)
    store_docs = {d.doc_id: d for d in store.list_documents()}
    enriched_docs = []
    for d in proj["documents"]:
        sd = store_docs.get(d["id"])
        if sd:
            enriched_docs.append({
                "doc_id": sd.doc_id,
                "filename": sd.filename,
                "file_type": sd.file_type,
                "size_bytes": sd.size_bytes,
                "sha256": sd.sha256,
                "page_count": sd.page_count,
                "chunk_count": sd.chunk_count,
                "ingested_at": sd.ingested_at,
            })
    proj["documents"] = enriched_docs
    return proj

@app.delete("/projects/{project_id}", tags=["projects"])
def delete_project(project_id: str) -> Dict[str, Any]:
    projects_db.delete_project(project_id)
    vector_store.delete_store(project_id)
    return {"status": "deleted", "id": project_id}

# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@app.post("/projects/{project_id}/documents", response_model=IngestOut, tags=["documents"], status_code=201)
async def upload_document(project_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ingest one document: extract -> chunk -> embed -> index.

    The upload is streamed to a temp file first so a large PDF never sits fully
    in memory, and so ingestion works on a real path (PyMuPDF wants one).

    Duplicate uploads (same SHA-256) are rejected with the existing ``doc_id``
    rather than indexed twice, which would over-weight that content at retrieval.
    """
    started = time.perf_counter()
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix or '(none)'}'. Supported: "
                f"{', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
            ),
        )

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            shutil.copyfileobj(file.file, handle)

        # Cheap duplicate check before paying for extraction and embedding.
        store = vector_store.get_store(project_id)
        digest = ingestion.compute_sha256(temp_path)
        existing = store.find_by_hash(digest)
        if existing is not None:
            projects_db.add_document(project_id, existing.doc_id, existing.filename)
            return {
                "doc_id": existing.doc_id,
                "filename": existing.filename,
                "chunk_count": existing.chunk_count,
                "page_count": existing.page_count,
                "duplicate_of": existing.doc_id,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }

        document = ingestion.ingest_file(temp_path, original_filename=filename)

        try:
            vectors = embeddings.embed_texts([chunk.text for chunk in document.chunks])
        except embeddings.EmbeddingModelUnavailable as exc:
            # Roll back the stored copy so a failed ingest leaves no orphan file.
            document.stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        store.add_document(document, vectors)
        projects_db.add_document(project_id, document.doc_id, document.filename)

        elapsed = int((time.perf_counter() - started) * 1000)
        audit_log.log_document_event(
            action="ingest",
            doc_id=document.doc_id,
            filename=document.filename,
            detail={
                "project_id": project_id,
                "chunk_count": len(document.chunks),
                "page_count": document.page_count,
                "sha256": document.sha256,
                "elapsed_ms": elapsed,
            },
        )
        return {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "chunk_count": len(document.chunks),
            "page_count": document.page_count,
            "duplicate_of": None,
            "elapsed_ms": elapsed,
        }

    except ingestion.UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ingestion.ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/documents", response_model=DocumentListOut, tags=["documents"])
def list_documents() -> Dict[str, Any]:
    """Every indexed document, newest first."""
    records = store.list_documents()
    return {
        "documents": [
            DocumentOut(
                doc_id=record.doc_id,
                filename=record.filename,
                file_type=record.file_type,
                size_bytes=record.size_bytes,
                sha256=record.sha256,
                page_count=record.page_count,
                chunk_count=record.chunk_count,
                ingested_at=record.ingested_at,
            )
            for record in records
        ],
        "total_chunks": sum(record.chunk_count for record in records),
    }


@app.delete("/projects/{project_id}/documents/{doc_id}", response_model=DeleteOut, tags=["documents"])
def delete_document(project_id: str, doc_id: str) -> Dict[str, Any]:
    """
    Remove a document, its vectors and its stored copy.

    Historic audit entries are intentionally **not** deleted: the log records
    that a query happened against that document, which is the whole point of an
    audit trail. Deleting the document removes it from future retrieval only.
    """
    store = vector_store.get_store(project_id)
    record = store.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id}")

    removed = store.delete_document(doc_id)
    projects_db.delete_document(doc_id)
    audit_log.log_document_event(
        action="delete",
        doc_id=doc_id,
        filename=record.filename,
        detail={"project_id": project_id, "removed_vectors": removed},
    )
    return {"doc_id": doc_id, "removed_vectors": removed}


@app.get("/projects/{project_id}/documents/{doc_id}/chunks", tags=["documents"])
def document_chunks(project_id: str, doc_id: str) -> Dict[str, Any]:
    """All chunks of one document, in reading order (useful for debugging)."""
    store = vector_store.get_store(project_id)
    if store.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id}")
    chunks = store.get_document_chunks(doc_id)
    return {
        "doc_id": doc_id,
        "chunks": [
            {
                "vector_id": chunk.vector_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


# --------------------------------------------------------------------------- #
# Query - the core RAG pipeline
# --------------------------------------------------------------------------- #
@app.post("/projects/{project_id}/query", response_model=QueryOut, tags=["query"])
def query(project_id: str, request: QueryIn) -> Dict[str, Any]:
    """
    Answer a question from the indexed documents.

    Pipeline, in order:

    1. embed the question locally (:mod:`app.embeddings`),
    2. retrieve the top-k most similar chunks above the similarity floor
       (:mod:`app.vector_store`),
    3. build a grounded prompt containing *only* those chunks and generate
       (:mod:`app.llm`),
    4. build the citation trail and mark which citations the answer used
       (:mod:`app.citations`),
    5. append the whole interaction to the local audit log (:mod:`app.audit_log`).

    Step 5 happens for every query, including refusals - an audit trail with gaps
    is not an audit trail.
    """
    started = time.perf_counter()

    try:
        query_vector = embeddings.embed_query(request.question)
    except embeddings.EmbeddingModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store = vector_store.get_store(project_id)
    results = store.search(
        query_vector,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )

    # Relevance check: if the best chunk is still below the threshold for grounding,
    # we treat it as an ungrounded query.
    grounded = False
    if results and results[0].score >= config.MIN_SIMILARITY_GROUNDED:
        grounded = True
    
    # If it's not grounded, and we're not in strict mode, we drop the retrieved chunks
    # so they don't pollute the context with irrelevant text for general chat.
    prompt_results = results if grounded else []

    generation = llm.generate(
        request.question, 
        prompt_results,
        strict_mode=request.strict_mode,
    )

    # Only show citations if the answer was grounded.
    if grounded:
        citation_list = citations_mod.build_citations(results)
        citations_mod.mark_referenced(citation_list, generation.answer)
    else:
        citation_list = []

    latency_ms = int((time.perf_counter() - started) * 1000)
    serialised = [citation.to_dict() for citation in citation_list]

    audit_id = audit_log.log_query(
        query=request.question,
        answer=generation.answer,
        top_k=request.top_k,
        retrieved_ids=[result.chunk.vector_id for result in results],
        citations=serialised,
        documents_touched=sorted({result.chunk.filename for result in results}),
        model=generation.model,
        latency_ms=latency_ms,
        embedding_model=embeddings.model_name(),
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        stub_mode=generation.stub,
        grounded=grounded,
    )

    projects_db.add_message(project_id, "user", request.question)
    projects_db.add_message(project_id, "assistant", generation.answer, serialised)

    return {
        "answer": generation.answer,
        "citations": serialised,
        "answered_from_context": grounded,
        "grounded": grounded,
        "stub_mode": generation.stub,
        "model": generation.model,
        "embedding_model": embeddings.model_name(),
        "top_k": request.top_k,
        "retrieved_count": len(results),
        "latency_ms": latency_ms,
        "audit_id": audit_id,
    }


@app.get("/projects/{project_id}/chunks/{vector_id}", response_model=ChunkOut, tags=["query"])
def get_chunk(project_id: str, vector_id: int) -> Dict[str, Any]:
    """
    Full, untruncated text behind a citation.

    This is what the UI calls when the user clicks a citation card - the moment
    where "the model claims X" becomes "here are the exact words it read".
    """
    store = vector_store.get_store(project_id)
    chunk = store.get_chunk(vector_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"No chunk with id {vector_id}")
    return {
        "vector_id": chunk.vector_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "location": citations_mod.format_location(
            chunk.filename, chunk.page_number, chunk.chunk_index
        ),
    }


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@app.get("/audit/queries", response_model=AuditListOut, tags=["audit"])
def audit_queries(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Recent audit entries, newest first."""
    entries = audit_log.get_recent_queries(limit=limit, offset=offset)
    return {"entries": entries, "total": audit_log.stats()["query_count"]}


@app.get("/audit/documents", tags=["audit"])
def audit_documents(limit: int = Query(default=100, ge=1, le=500)) -> Dict[str, Any]:
    """Document ingest/delete history."""
    return {"events": audit_log.get_document_events(limit=limit)}


@app.get("/audit/verify", response_model=AuditVerifyOut, tags=["audit"])
def audit_verify() -> Dict[str, Any]:
    """Re-compute the audit hash chain and report any tampering."""
    return audit_log.verify_chain()


@app.post("/audit/export", tags=["audit"])
def audit_export() -> Dict[str, Any]:
    """
    Export the audit log to JSON Lines next to the database.

    Writes to disk only - there is no upload path, by design.
    """
    destination = config.LOGS_DIR / "audit_export.jsonl"
    written = audit_log.export_jsonl(destination)
    return {"path": str(destination), "records": written}


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the API with uvicorn bound to loopback."""
    import uvicorn

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        # Reload is deliberately off: the Tauri shell owns this process's
        # lifecycle and a reloader would orphan child processes on exit.
        reload=False,
    )


if __name__ == "__main__":
    main()
