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
Everything under ``/projects/{id}`` 404s for an unknown project id.

=================================================  ===================================
``GET  /health``                                   full status blob
``GET  /health/offline-check``                     offline badge data
``POST /projects`` / ``GET /projects``             create / list projects
``GET  /projects/{id}``                            documents + chat history
``DELETE /projects/{id}``                          remove a project and its files
``POST /projects/{id}/documents``                  upload + ingest + index one file
``DELETE /projects/{id}/documents/{doc_id}``       remove a document and its vectors
``GET  /projects/{id}/documents/{doc_id}/chunks``  every chunk of one document
``POST /projects/{id}/query``                      ask a question, get answer + citations
``GET  /projects/{id}/chunks/{vector_id}``         untruncated source text for a citation
``GET  /audit/queries``                            recent audit entries
``GET  /audit/verify``                             verify the audit hash chain
``POST /audit/export``                             dump the audit log to JSON Lines
``/setup/...``                                     model catalog, download, switch
=================================================  ===================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

# 1. config first - it sets the offline environment variables on import.
from app import config

# 2. network guard second - patch sockets before anything else can use them.
from app import network_guard

network_guard.install_guard()

# 3. Everything else may now be imported safely.
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import audit_log, citations as citations_mod, embeddings, ingestion, llm, projects_db, references, setup, vector_store
from app.schemas import (
    AuditListOut,
    AuditVerifyOut,
    ChunkOut,
    DeleteOut,
    HealthOut,
    IngestOut,
    OfflineCheckOut,
    QueryIn,
    QueryOut,
)
from app.vector_store import DocumentRecord, SearchResult, VectorStoreError

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

    # Load the model chosen in the setup wizard / model switcher, not the default.
    setup.apply_saved_model()
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
        # Per-project counts come from GET /projects/{id}; the embedding model
        # is shared by every project.
        "index": {"embedding_model": embeddings.model_name()},
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

def _require_project(project_id: str) -> None:
    """
    404 for unknown project ids.

    The id becomes a directory name under PROJECTS_DIR, so only ids this app
    created (UUIDs in the projects table) may reach the filesystem.
    """
    if not projects_db.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

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
    _require_project(project_id)
    projects_db.delete_project(project_id)
    removed = vector_store.delete_store(project_id)
    # Keep the ingest/delete history complete, as a single-document delete would.
    for record in removed:
        audit_log.log_document_event(
            action="delete",
            doc_id=record.doc_id,
            filename=record.filename,
            detail={"project_id": project_id, "reason": "project deleted"},
        )
    return {"status": "deleted", "id": project_id}

# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@app.post("/projects/{project_id}/documents", response_model=IngestOut, tags=["documents"], status_code=201)
def upload_document(project_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ingest one document: extract -> chunk -> embed -> index.

    A plain ``def`` on purpose: FastAPI runs it on a worker thread. As an
    ``async def`` the extraction and embedding ran on the event loop and every
    other request (badge, project list, questions) waited for the whole ingest.

    The upload is streamed to a temp file first so a large PDF never sits fully
    in memory, and so ingestion works on a real path (PyMuPDF wants one).

    Duplicate uploads (same SHA-256) are rejected with the existing ``doc_id``
    rather than indexed twice, which would over-weight that content at retrieval.
    """
    _require_project(project_id)
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
        file.file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.delete("/projects/{project_id}/documents/{doc_id}", response_model=DeleteOut, tags=["documents"])
def delete_document(project_id: str, doc_id: str) -> Dict[str, Any]:
    """
    Remove a document, its vectors and its stored copy.

    Historic audit entries are intentionally **not** deleted: the log records
    that a query happened against that document, which is the whole point of an
    audit trail. Deleting the document removes it from future retrieval only.
    """
    _require_project(project_id)
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
    _require_project(project_id)
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
def _opening_passages(
    results: List[SearchResult], targets: List[DocumentRecord]
) -> List[SearchResult]:
    """
    The first passages of each target document, in reading order.

    Two for a single document (enough to say what it is about), one each when
    several are named - every passage costs seconds of reading on a CPU.
    """
    per_doc = 2 if len(targets) == 1 else 1
    opening: List[SearchResult] = []
    for doc in targets:
        in_order = sorted(
            (r for r in results if r.chunk.doc_id == doc.doc_id),
            key=lambda r: r.chunk.chunk_index,
        )
        opening.extend(in_order[:per_doc])
    return opening[:3]


@dataclass
class _PreparedQuery:
    """Everything decided before generation; shared by /query and /query/stream."""

    started: float
    results: List[SearchResult]          # what retrieval considered (audit log)
    prompt_results: List[SearchResult]   # what the model is given (citations)
    grounded: bool
    history: List[Dict[str, str]]
    documents: List[str]
    t_embed: float
    t_retrieve: float


def _prepare_query(project_id: str, request: QueryIn) -> _PreparedQuery:
    """Steps 1-2 of the pipeline: embed, retrieve, choose the passages."""
    started = time.perf_counter()
    logger.info("Received query. Strict mode: %s", request.strict_mode)

    # 1. Embedding
    t0 = time.perf_counter()
    try:
        query_vector = embeddings.embed_query(request.question)
    except embeddings.EmbeddingModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    t_embed = time.perf_counter() - t0

    # 2. Retrieval
    t0 = time.perf_counter()
    store = vector_store.get_store(project_id)
    documents = store.list_documents()
    # A question that names a file ("what's in resume.pdf?") or points at one
    # ("what is this doc about?") is about that document, whatever it matches.
    targets = references.referred_documents(request.question, documents)
    if targets:
        results = store.search_documents(query_vector, [doc.doc_id for doc in targets])
    else:
        results = store.search(
            query_vector,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
    # A passage that is only a slide heading matches almost any question.
    results = [r for r in results if len(r.chunk.text.split()) >= config.MIN_PASSAGE_WORDS]
    t_retrieve = time.perf_counter() - t0

    # Relevance check: if the best chunk is still below the threshold for grounding,
    # we treat it as an ungrounded query - unless the question is about a named
    # document, which is grounded in that document by definition.
    best = results[0].score if results else 0.0
    grounded = bool(results) and (bool(targets) or best >= config.MIN_SIMILARITY_GROUNDED)

    # If it's not grounded, and we're not in strict mode, we drop the retrieved chunks
    # so they don't pollute the context with irrelevant text for general chat.
    # When it is grounded, only passages close to the best match go to the model:
    # reading each passage costs seconds on a CPU.
    if not grounded:
        prompt_results = []
    elif targets and best < config.MIN_SIMILARITY_GROUNDED:
        # A vague question about the document ("what is it about?") matches no
        # passage well, so give the model the document's opening instead.
        prompt_results = _opening_passages(results, targets)
    else:
        prompt_results = [
            r for r in results if r.score >= best - config.RELEVANCE_MARGIN
        ][: request.top_k]
    if targets:
        # Searching a named document scores all its chunks; keep the log to the
        # best few plus whatever the model was given.
        shown = {r.chunk.vector_id for r in prompt_results}
        results = [r for i, r in enumerate(results) if i < request.top_k or r.chunk.vector_id in shown]

    # The last two exchanges are enough for follow-ups; more history only
    # makes every answer slower (see llm.generate).
    history = projects_db.get_recent_messages(project_id, limit=4)

    return _PreparedQuery(
        started=started,
        results=results,
        prompt_results=prompt_results,
        grounded=grounded,
        history=history,
        documents=[doc.filename for doc in documents],
        t_embed=t_embed,
        t_retrieve=t_retrieve,
    )


def _generate(
    request: QueryIn, prepared: _PreparedQuery, **streaming: Any
) -> llm.GenerationResult:
    """Step 3: generate. ``streaming`` is passed through (on_token, should_stop)."""
    return llm.generate(
        request.question,
        prepared.prompt_results,
        strict_mode=request.strict_mode,
        history=prepared.history,
        documents=prepared.documents,
        **streaming,
    )


def _finish_query(
    project_id: str,
    request: QueryIn,
    prepared: _PreparedQuery,
    generation: llm.GenerationResult,
) -> Dict[str, Any]:
    """Steps 4-5: citations, audit entry, chat history. Returns the response body."""
    started, results, prompt_results, grounded = (
        prepared.started, prepared.results, prepared.prompt_results, prepared.grounded
    )
    t_embed, t_retrieve = prepared.t_embed, prepared.t_retrieve

    # Citations are exactly the passages the model was given, numbered as in the
    # prompt, so [n] in the answer is citation n. Empty when not grounded.
    citation_list = citations_mod.build_citations(prompt_results)
    citations_mod.mark_referenced(citation_list, generation.answer)

    latency_ms = int((time.perf_counter() - started) * 1000)
    serialised = [citation.to_dict() for citation in citation_list]
    
    logger.info(
        "Query timing breakdown: embed=%.3fs, retrieve=%.3fs, assembly=%.3fs, prefill=%.3fs, generate=%.3fs, total=%.3fs",
        t_embed, t_retrieve, generation.t_assembly, generation.t_prefill, generation.t_generation, latency_ms / 1000.0
    )

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
        project_id=project_id,
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
        # Passages the answer was built from; the audit log keeps every retrieved id.
        "retrieved_count": len(prompt_results),
        "latency_ms": latency_ms,
        "audit_id": audit_id,
    }


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
    _require_project(project_id)
    prepared = _prepare_query(project_id, request)
    return _finish_query(project_id, request, prepared, _generate(request, prepared))


#: While nothing else is being sent, the stream says "still here" this often, so
#: the UI can tell a slow prompt read from a backend that has gone away.
_STREAM_PING_SECONDS = 3.0
#: If nothing has read the stream for this long, the client is gone even if the
#: server was never told; stop generating rather than keep the model busy.
_STREAM_ABANDONED_SECONDS = 10.0


@app.post("/projects/{project_id}/query/stream", tags=["query"])
async def query_stream(project_id: str, request: QueryIn, http_request: Request) -> StreamingResponse:
    """
    Same as ``POST /query``, but the answer arrives while it is being written.

    The body is newline-delimited JSON, one event per line::

        {"type": "token", "text": "..."}   the next piece of the answer
        {"type": "ping"}                   keep-alive while nothing else is sent
        {"type": "done", ...}              the full /query response; its answer is final
        {"type": "error", "detail": "..."}

    The answer is written on its own thread. If the client goes away mid-answer
    (Stop button, project switch, window closed), generation stops at the next
    token - so the next question never waits behind an answer nobody is reading -
    and what was written is still logged, marked as stopped.
    """
    await run_in_threadpool(_require_project, project_id)
    # Before the stream starts, so e.g. a missing embedding model is a plain 503.
    prepared = await run_in_threadpool(_prepare_query, project_id, request)

    events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    stop = threading.Event()
    pulled_at = time.monotonic()

    def should_stop() -> bool:
        # Two independent signals, so a lost disconnect notice can't leave the
        # model writing for nobody: an explicit stop, or the stream going unread.
        return stop.is_set() or time.monotonic() - pulled_at > _STREAM_ABANDONED_SECONDS

    def write_answer() -> None:
        try:
            generation = _generate(
                request,
                prepared,
                on_token=lambda text: events.put({"type": "token", "text": text}),
                should_stop=should_stop,
            )
            events.put({"type": "done", **_finish_query(project_id, request, prepared, generation)})
        except Exception as exc:  # noqa: BLE001 - the client must always hear how it ended
            logger.exception("Streamed answer failed")
            events.put({"type": "error", "detail": str(exc)})

    threading.Thread(target=write_answer, name="safe-insight-answer", daemon=True).start()

    async def event_lines():
        nonlocal pulled_at
        last_sent = time.monotonic()
        try:
            while True:
                pulled_at = time.monotonic()
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    if await http_request.is_disconnected():
                        return
                    if time.monotonic() - last_sent >= _STREAM_PING_SECONDS:
                        yield b'{"type": "ping"}\n'
                        last_sent = time.monotonic()
                    await asyncio.sleep(0.05)
                    continue
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
                last_sent = time.monotonic()
                if event["type"] in ("done", "error"):
                    return
        finally:
            # Finished, or the client left: either way the writer can stop now.
            stop.set()

    return StreamingResponse(event_lines(), media_type="application/x-ndjson")


@app.get("/projects/{project_id}/chunks/{vector_id}", response_model=ChunkOut, tags=["query"])
def get_chunk(project_id: str, vector_id: int) -> Dict[str, Any]:
    """
    Full, untruncated text behind a citation.

    This is what the UI calls when the user clicks a citation card - the moment
    where "the model claims X" becomes "here are the exact words it read".
    """
    _require_project(project_id)
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
