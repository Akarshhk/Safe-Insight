"""
Local embedding model wrapper.

Loads a sentence-transformers model **from the local cache only** and turns text
into L2-normalised float32 vectors. Normalising here rather than in the vector
store means a plain FAISS inner-product index computes cosine similarity, which
keeps :mod:`app.vector_store` simple and its scores directly interpretable
(1.0 = identical, 0.0 = unrelated).

Model choice
------------
Default is ``BAAI/bge-small-en-v1.5`` (384 dims, ~130 MB): strong retrieval
quality for its size and comfortable on an 8 GB CPU-only laptop. If it is not in
the cache we fall back to ``all-MiniLM-L6-v2`` (also 384 dims), so a partial
download still gives a working app.

Both are 384-dimensional, which matters: the FAISS index is built for a fixed
dimension. Swapping to a model with a different dimension requires deleting and
rebuilding the index - :mod:`app.vector_store` detects the mismatch and says so.

Threading
---------
The model is loaded lazily behind a lock. FastAPI may serve concurrent requests
and ``SentenceTransformer.encode`` is not guaranteed re-entrant during load, so
first use is serialised; afterwards encode calls proceed in parallel.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_model = None                      # type: ignore[var-annotated]
_model_name: Optional[str] = None
_lock = threading.Lock()


class EmbeddingModelUnavailable(RuntimeError):
    """
    Raised when no embedding model can be loaded from the local cache.

    Almost always means ``python download_model.py`` has not been run yet. The
    message is user-facing: the UI shows it verbatim.
    """


def _load_model():
    """
    Load the primary model, falling back to the secondary one.

    ``local_files_only=True`` is what enforces "no download at runtime" - with
    the offline env vars from :mod:`app.config` it is belt *and* braces.
    """
    from sentence_transformers import SentenceTransformer  # imported lazily: ~3 s

    attempts = [config.EMBEDDING_MODEL_NAME, config.EMBEDDING_MODEL_FALLBACK]
    errors: List[str] = []

    for name in attempts:
        try:
            logger.info("Loading embedding model %s from local cache", name)
            model = SentenceTransformer(
                name,
                cache_folder=str(config.EMBEDDING_CACHE_DIR),
                local_files_only=True,
                device="cpu",
            )
            return model, name
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next one"
            errors.append(f"{name}: {exc}")
            logger.warning("Could not load embedding model %s (%s)", name, exc)

    raise EmbeddingModelUnavailable(
        "No local embedding model found. Run 'python download_model.py' once "
        "while online to populate the cache at "
        f"{config.EMBEDDING_CACHE_DIR}.\nAttempts:\n  - " + "\n  - ".join(errors)
    )


def get_model():
    """Return the loaded model, loading it on first call (thread-safe)."""
    global _model, _model_name
    if _model is None:
        with _lock:
            if _model is None:  # re-check inside the lock
                _model, _model_name = _load_model()
    return _model


def is_loaded() -> bool:
    """True if the model is already in memory (used by ``/health``)."""
    return _model is not None


def model_name() -> Optional[str]:
    """Name of the model actually loaded, or ``None`` before first use."""
    return _model_name


def dimension() -> int:
    """
    Embedding dimension of the loaded model.

    :mod:`app.vector_store` calls this to size a new FAISS index and to validate
    an index loaded from disk.
    """
    return int(get_model().get_sentence_embedding_dimension())


def _uses_bge_prefix() -> bool:
    """BGE v1.5 models expect an instruction prefix on queries; MiniLM does not."""
    return "bge" in (model_name() or "").lower()


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a batch of passages (document chunks).

    Args:
        texts: chunk texts, in the order the caller wants the rows returned.
        batch_size: kept small by default; 32 keeps peak RAM modest on 8 GB
            machines while still amortising model overhead.

    Returns:
        ``(len(texts), dimension())`` float32 array of unit-length vectors.
        An empty input returns a correctly shaped empty array so callers can
        ``np.vstack`` without special-casing.
    """
    if not texts:
        return np.zeros((0, dimension()), dtype=np.float32)

    vectors = get_model().encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user question.

    Returns a ``(1, dimension())`` array - the shape FAISS ``search`` wants - so
    callers never have to reshape.
    """
    text = config.BGE_QUERY_PREFIX + query if _uses_bge_prefix() else query
    vector = get_model().encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vector, dtype=np.float32)


def warm_up() -> None:
    """
    Load the model and run one throwaway encode.

    Called at startup so the first real question is not 3-5 seconds slower than
    every subsequent one. Failures are logged, not raised: the app should still
    boot and show a clear error in the UI rather than refusing to start.
    """
    try:
        embed_texts(["warm up"])
        logger.info("Embedding model ready: %s (dim=%d)", model_name(), dimension())
    except EmbeddingModelUnavailable as exc:
        logger.error("Embedding model unavailable: %s", exc)


__all__ = [
    "EmbeddingModelUnavailable",
    "get_model",
    "is_loaded",
    "model_name",
    "dimension",
    "embed_texts",
    "embed_query",
    "warm_up",
]
