"""
Persistent FAISS vector store with a JSON metadata sidecar.

Responsibilities
----------------
* hold the chunk vectors and search them by cosine similarity,
* remember, for every vector, which document / page / chunk it came from,
* survive an app restart without re-indexing,
* support incremental add and per-document delete.

Index choice
------------
``IndexIDMap2(IndexFlatIP(dim))``.

* ``IndexFlatIP`` is an exact, brute-force inner-product index. Because
  :mod:`app.embeddings` returns unit vectors, inner product *is* cosine
  similarity. Exact search is the right call at this scale: a few hundred
  documents is tens of thousands of vectors, where a flat scan is
  sub-millisecond and an approximate index (IVF/HNSW) would add training,
  tuning, and recall loss for no user-visible gain.
* ``IndexIDMap2`` lets us attach our own stable 64-bit ids and, crucially,
  supports ``remove_ids`` - that is what makes "remove this document" possible
  without rebuilding the whole index.

Why a JSON sidecar rather than a second database?
    FAISS stores vectors and ids, nothing else. The metadata is small (a few KB
    per document), human-readable - which matters when the examiner asks to see
    the citation trail - and trivially diffable. SQLite is reserved for the audit
    log, where append-heavy writes and durability actually matter.

Concurrency
-----------
All public methods take a re-entrant lock. FAISS index objects are not
thread-safe for concurrent write + read, and FastAPI will happily overlap an
upload with a query.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app import config
from app.ingestion import IngestedDocument

logger = logging.getLogger(__name__)

#: Bump when the on-disk metadata shape changes so old state fails loudly.
METADATA_VERSION = 1


class VectorStoreError(RuntimeError):
    """Raised for unrecoverable index/metadata problems (e.g. dimension mismatch)."""


@dataclass
class ChunkRecord:
    """Everything known about one indexed chunk. Serialised into metadata.json."""

    vector_id: int
    doc_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int]
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class DocumentRecord:
    """Registry entry for one ingested document."""

    doc_id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    page_count: Optional[int]
    chunk_count: int
    ingested_at: str
    stored_path: str


@dataclass
class SearchResult:
    """One retrieved chunk plus its similarity score, ordered best-first."""

    chunk: ChunkRecord
    score: float
    rank: int = 0


class VectorStore:
    """
    File-backed FAISS store. Instantiate once per process (see :data:`store`).

    Typical lifecycle::

        store.load()                       # at startup
        store.add_document(doc, vectors)   # on upload
        store.search(query_vector, k=5)    # on question
        store.delete_document(doc_id)      # on delete
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._lock = threading.RLock()

        self._index = None                       # faiss.IndexIDMap2 | None
        self._dimension: Optional[int] = None
        self._embedding_model: Optional[str] = None
        self._next_vector_id: int = 0
        self._chunks: Dict[int, ChunkRecord] = {}
        self._documents: Dict[str, DocumentRecord] = {}
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _new_index(self, dimension: int):
        """Create an empty id-mapped flat inner-product index."""
        import faiss

        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def load(self, dimension: Optional[int] = None, embedding_model: Optional[str] = None) -> None:
        """
        Load index + metadata from disk, or start an empty store.

        Args:
            dimension: expected embedding dimension. When provided and the
                persisted index disagrees, we raise rather than silently
                producing garbage similarity scores.
            embedding_model: name of the model in use, recorded on first save so
                a later mismatch can be reported to the user.

        This never raises for "no index yet" - a fresh install is normal.
        """
        import faiss

        with self._lock:
            config.ensure_directories()

            if self.metadata_path.exists():
                self._load_metadata()

            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                logger.info(
                    "Loaded FAISS index: %d vectors, %d documents",
                    self._index.ntotal,
                    len(self._documents),
                )
            else:
                if dimension is None:
                    # Defer creation until the first add, when the embedding
                    # model (and therefore the dimension) is definitely known.
                    self._loaded = True
                    return
                self._index = self._new_index(dimension)
                self._dimension = dimension

            if dimension is not None and self._index is not None:
                if self._index.d != dimension:
                    raise VectorStoreError(
                        f"Index dimension mismatch: index on disk is {self._index.d}-d "
                        f"but the loaded embedding model produces {dimension}-d vectors. "
                        f"The embedding model changed. Delete {self.index_path.parent} "
                        f"and re-ingest your documents."
                    )
                self._dimension = dimension

            if embedding_model:
                self._embedding_model = self._embedding_model or embedding_model

            self._loaded = True

    def _load_metadata(self) -> None:
        """Read metadata.json into the in-memory registries."""
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(
                f"Corrupt metadata file at {self.metadata_path}: {exc}. "
                f"Delete the index directory and re-ingest."
            ) from exc

        version = payload.get("version")
        if version != METADATA_VERSION:
            raise VectorStoreError(
                f"Metadata version {version} is not supported by this build "
                f"(expected {METADATA_VERSION}). Delete {self.metadata_path.parent} "
                f"and re-ingest your documents."
            )

        self._dimension = payload.get("dimension")
        self._embedding_model = payload.get("embedding_model")
        self._next_vector_id = int(payload.get("next_vector_id", 0))
        self._chunks = {
            int(vector_id): ChunkRecord(**record)
            for vector_id, record in payload.get("chunks", {}).items()
        }
        self._documents = {
            doc_id: DocumentRecord(**record)
            for doc_id, record in payload.get("documents", {}).items()
        }

    def save(self) -> None:
        """
        Persist index and metadata atomically.

        Both files are written to a ``.tmp`` sibling and then ``os.replace``d, so
        a crash mid-write leaves the previous good state intact rather than a
        truncated index the app cannot open.
        """
        import faiss

        with self._lock:
            if self._index is None:
                return
            config.ensure_directories()

            tmp_index = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            tmp_index.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(tmp_index))
            os.replace(tmp_index, self.index_path)

            payload = {
                "version": METADATA_VERSION,
                "dimension": self._dimension,
                "embedding_model": self._embedding_model,
                "next_vector_id": self._next_vector_id,
                "documents": {k: asdict(v) for k, v in self._documents.items()},
                "chunks": {str(k): asdict(v) for k, v in self._chunks.items()},
            }
            tmp_meta = self.metadata_path.with_suffix(".json.tmp")
            tmp_meta.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_meta, self.metadata_path)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add_document(self, document: IngestedDocument, vectors: np.ndarray) -> List[int]:
        """
        Add one document's chunks to the index (incremental - nothing is rebuilt).

        Args:
            document: the ingested document, with ``chunks`` populated.
            vectors: ``(len(document.chunks), dim)`` float32 unit vectors, in the
                same order as ``document.chunks``.

        Returns:
            The vector ids assigned, in chunk order.

        Raises:
            VectorStoreError: on a row-count or dimension mismatch, or if the
                document id is already indexed.
        """
        with self._lock:
            if document.doc_id in self._documents:
                raise VectorStoreError(
                    f"Document {document.doc_id} is already indexed. Delete it first."
                )
            if vectors.ndim != 2 or vectors.shape[0] != len(document.chunks):
                raise VectorStoreError(
                    f"Expected {len(document.chunks)} vectors, got shape {vectors.shape}."
                )

            dimension = int(vectors.shape[1])
            if self._index is None:
                self._index = self._new_index(dimension)
                self._dimension = dimension
            elif self._index.d != dimension:
                raise VectorStoreError(
                    f"Cannot add {dimension}-d vectors to a {self._index.d}-d index."
                )

            vector_ids = [self._next_vector_id + offset for offset in range(len(document.chunks))]
            self._next_vector_id += len(document.chunks)

            self._index.add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32),
                np.asarray(vector_ids, dtype=np.int64),
            )

            for vector_id, chunk in zip(vector_ids, document.chunks):
                self._chunks[vector_id] = ChunkRecord(
                    vector_id=vector_id,
                    doc_id=document.doc_id,
                    filename=document.filename,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )

            self._documents[document.doc_id] = DocumentRecord(
                doc_id=document.doc_id,
                filename=document.filename,
                file_type=document.file_type,
                size_bytes=document.size_bytes,
                sha256=document.sha256,
                page_count=document.page_count,
                chunk_count=len(document.chunks),
                ingested_at=document.ingested_at,
                stored_path=str(document.stored_path),
            )

            self.save()
            logger.info(
                "Indexed %s: %d chunks (index now holds %d vectors)",
                document.filename,
                len(document.chunks),
                self._index.ntotal,
            )
            return vector_ids

    def delete_document(self, doc_id: str, remove_file: bool = True) -> int:
        """
        Remove a document's vectors, metadata and (optionally) its stored file.

        Returns:
            Number of vectors removed. 0 means the document was not indexed.
        """
        with self._lock:
            record = self._documents.get(doc_id)
            if record is None:
                return 0

            vector_ids = [
                vector_id
                for vector_id, chunk in self._chunks.items()
                if chunk.doc_id == doc_id
            ]
            if vector_ids and self._index is not None:
                # FAISS's Python wrapper turns a 1-D int64 array into an
                # IDSelectorBatch for us; this form works across faiss versions.
                self._index.remove_ids(np.asarray(vector_ids, dtype=np.int64))

            for vector_id in vector_ids:
                self._chunks.pop(vector_id, None)
            self._documents.pop(doc_id, None)

            if remove_file:
                try:
                    Path(record.stored_path).unlink(missing_ok=True)
                except OSError as exc:  # pragma: no cover - filesystem edge case
                    logger.warning("Could not delete %s: %s", record.stored_path, exc)

            self.save()
            logger.info("Deleted %s (%d vectors)", record.filename, len(vector_ids))
            return len(vector_ids)

    def clear(self) -> None:
        """Drop everything. Used by the 'reset' endpoint and by tests."""
        with self._lock:
            self._index = self._new_index(self._dimension) if self._dimension else None
            self._chunks.clear()
            self._documents.clear()
            self._next_vector_id = 0
            self.save()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Retrieve the ``top_k`` most similar chunks.

        Args:
            query_vector: ``(1, dim)`` unit vector from
                :func:`app.embeddings.embed_query`.
            top_k: how many chunks to return, before the similarity filter.
            min_similarity: cosine floor; defaults to ``config.MIN_SIMILARITY``.
                Filtering here is what lets the answer say "not in your
                documents" instead of inventing one from unrelated text.

        Returns:
            Best-first results. Empty when the index is empty or nothing clears
            the floor.
        """
        floor = config.MIN_SIMILARITY if min_similarity is None else min_similarity

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []

            k = max(1, min(top_k, self._index.ntotal))
            scores, ids = self._index.search(
                np.ascontiguousarray(query_vector, dtype=np.float32), k
            )

            results: List[SearchResult] = []
            for score, vector_id in zip(scores[0], ids[0]):
                if vector_id == -1:          # FAISS pads short result sets with -1
                    continue
                if float(score) < floor:
                    continue
                chunk = self._chunks.get(int(vector_id))
                if chunk is None:            # metadata/index drift; skip defensively
                    logger.warning("Vector %s has no metadata record", vector_id)
                    continue
                results.append(SearchResult(chunk=chunk, score=float(score)))

            for rank, result in enumerate(results, start=1):
                result.rank = rank
            return results

    def get_chunk(self, vector_id: int) -> Optional[ChunkRecord]:
        """Look up one chunk by vector id - backs the 'show me the source' click."""
        with self._lock:
            return self._chunks.get(int(vector_id))

    def get_document_chunks(self, doc_id: str) -> List[ChunkRecord]:
        """All chunks of one document, in reading order."""
        with self._lock:
            chunks = [c for c in self._chunks.values() if c.doc_id == doc_id]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def list_documents(self) -> List[DocumentRecord]:
        """Every indexed document, newest first."""
        with self._lock:
            return sorted(
                self._documents.values(), key=lambda d: d.ingested_at, reverse=True
            )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """One document record, or ``None``."""
        with self._lock:
            return self._documents.get(doc_id)

    def find_by_hash(self, sha256: str) -> Optional[DocumentRecord]:
        """Find an already-indexed document with identical content."""
        with self._lock:
            for record in self._documents.values():
                if record.sha256 == sha256:
                    return record
        return None

    def stats(self) -> Dict[str, Any]:
        """Counts and configuration, surfaced by ``GET /health`` and the UI."""
        with self._lock:
            return {
                "documents": len(self._documents),
                "chunks": len(self._chunks),
                "vectors": int(self._index.ntotal) if self._index is not None else 0,
                "dimension": self._dimension,
                "embedding_model": self._embedding_model,
                "index_path": str(self.index_path),
                "loaded": self._loaded,
            }


_stores = {}
_stores_lock = threading.Lock()

def get_store(project_id: str) -> VectorStore:
    """Get or create the vector store instance for a specific project."""
    with _stores_lock:
        if project_id not in _stores:
            project_dir = config.PROJECTS_DIR / project_id
            index_path = project_dir / "index.faiss"
            metadata_path = project_dir / "metadata.json"
            store = VectorStore(index_path=index_path, metadata_path=metadata_path)
            store.load() # load dimension if available
            _stores[project_id] = store
        return _stores[project_id]

def delete_store(project_id: str) -> None:
    """Drop the store from memory and disk."""
    with _stores_lock:
        store = _stores.pop(project_id, None)
        if store:
            store.clear()
        
        project_dir = config.PROJECTS_DIR / project_id
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)

__all__ = [
    "VectorStore",
    "VectorStoreError",
    "ChunkRecord",
    "DocumentRecord",
    "SearchResult",
    "get_store",
    "delete_store",
]
