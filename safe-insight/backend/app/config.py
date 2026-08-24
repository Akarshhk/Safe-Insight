"""
Central configuration for the Safe Insight backend.

Every tunable knob (model paths, chunk sizes, top-k, data directories) lives
here so that no other module hard-codes a path or a magic number. Values can be
overridden with environment variables, which is how the Tauri shell customises
the backend when it spawns it.

Design note for the next agent picking this up:
    Import this module FIRST in any entrypoint. Its side effect of setting the
    HuggingFace "offline" environment variables must happen before
    ``sentence_transformers`` / ``transformers`` are imported, otherwise those
    libraries may attempt a network call to check for model updates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, List, Set
import platformdirs

# --------------------------------------------------------------------------- #
# Offline enforcement.
# Must be set before transformers / sentence-transformers are imported anywhere
# in the process, hence the module-level side effect.
# --------------------------------------------------------------------------- #
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _env_str(key: str, default: str) -> str:
    """Read a string env var, treating empty/whitespace as 'unset'."""
    value = os.environ.get(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on anything unparseable."""
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on anything unparseable."""
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# backend/app/config.py  ->  backend/app  ->  backend  ->  safe-insight/
APP_DIR: Final[Path] = Path(__file__).resolve().parent
BACKEND_DIR: Final[Path] = APP_DIR.parent
PROJECT_ROOT: Final[Path] = BACKEND_DIR.parent

#: Everything the app writes at runtime lives under this one directory, so the
#: user can wipe all local state by deleting a single folder.
_default_data_dir = platformdirs.user_data_dir("safe-insight", False)
DATA_DIR: Final[Path] = Path(_env_str("SAFE_INSIGHT_DATA_DIR", _default_data_dir))
MODELS_DIR: Final[Path] = Path(_env_str("SAFE_INSIGHT_MODELS_DIR", str(DATA_DIR / "models")))

UPLOADS_DIR: Final[Path] = DATA_DIR / "uploads"   # original uploaded files
PROJECTS_DIR: Final[Path] = DATA_DIR / "projects" # project-specific indices
LOGS_DIR: Final[Path] = DATA_DIR / "logs"         # audit database

INDEX_DIR: Final[Path] = DATA_DIR / "index"       # legacy global index
FAISS_INDEX_PATH: Final[Path] = INDEX_DIR / "safe_insight.faiss"
METADATA_PATH: Final[Path] = INDEX_DIR / "metadata.json"

AUDIT_DB_PATH: Final[Path] = LOGS_DIR / "audit.sqlite3"
APP_DB_PATH: Final[Path] = DATA_DIR / "app_state.sqlite3"

#: Local cache for the sentence-transformers embedding model. Populated once by
#: ``download_model.py`` while online; read-only offline afterwards.
EMBEDDING_CACHE_DIR: Final[Path] = MODELS_DIR / "embeddings"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(EMBEDDING_CACHE_DIR)
os.environ.setdefault("HF_HOME", str(EMBEDDING_CACHE_DIR))

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
#: Default generation model. Swap to "Llama-3.2-3B-Instruct-Q4_K_M.gguf" (or any
#: other GGUF) by setting SAFE_INSIGHT_LLM_FILE, or by editing this line.
LLM_MODEL_FILENAME: str = _env_str(
    "SAFE_INSIGHT_LLM_FILE", "Phi-3.5-mini-instruct-Q4_K_M.gguf"
)
LLM_MODEL_PATH: Path = Path(
    _env_str("SAFE_INSIGHT_LLM_PATH", str(MODELS_DIR / "llm" / LLM_MODEL_FILENAME))
)

#: Primary embedding model; the fallback is used when the primary is missing
#: from the local cache (e.g. the user only downloaded the smaller MiniLM).
EMBEDDING_MODEL_NAME: Final[str] = _env_str(
    "SAFE_INSIGHT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
EMBEDDING_MODEL_FALLBACK: Final[str] = _env_str(
    "SAFE_INSIGHT_EMBEDDING_MODEL_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2"
)

#: BGE models are trained with an instruction prefix on the *query* side only.
#: ``embeddings.py`` applies it only when the loaded model looks like a BGE model.
BGE_QUERY_PREFIX: Final[str] = (
    "Represent this sentence for searching relevant passages: "
)

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE_TOKENS: Final[int] = _env_int("SAFE_INSIGHT_CHUNK_SIZE", 500)
CHUNK_OVERLAP_TOKENS: Final[int] = _env_int("SAFE_INSIGHT_CHUNK_OVERLAP", 50)
#: Rough words-per-token ratio for English prose, so chunking never needs a
#: tokenizer download. ~0.75 words per token is the usual rule of thumb.
WORDS_PER_TOKEN: Final[float] = _env_float("SAFE_INSIGHT_WORDS_PER_TOKEN", 0.75)

# --------------------------------------------------------------------------- #
# Retrieval + generation
# --------------------------------------------------------------------------- #
DEFAULT_TOP_K: Final[int] = _env_int("SAFE_INSIGHT_TOP_K", 5)
MAX_TOP_K: Final[int] = _env_int("SAFE_INSIGHT_MAX_TOP_K", 20)
#: Cosine-similarity floor. Chunks below this are dropped before they reach the
#: prompt, so an off-topic question yields "not in your documents" rather than a
#: confident answer built on irrelevant context.
MIN_SIMILARITY: Final[float] = _env_float("SAFE_INSIGHT_MIN_SIMILARITY", 0.20)

#: Cutoff for grounded vs general answers. If the best retrieved chunk scores below
#: this, the question is treated as a general/small-talk question.
MIN_SIMILARITY_GROUNDED: Final[float] = _env_float("SAFE_INSIGHT_MIN_SIMILARITY_GROUNDED", 0.50)

def _get_llm_context_tokens() -> int:
    try:
        import json
        catalog_path = APP_DIR / "models_catalog.json"
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        for model in catalog:
            if model.get("filename") == LLM_MODEL_FILENAME:
                return model.get("n_ctx", 4096)
    except Exception:
        pass
    return 4096

LLM_CONTEXT_TOKENS: int = _env_int("SAFE_INSIGHT_LLM_CTX", _get_llm_context_tokens())
LLM_MAX_OUTPUT_TOKENS: Final[int] = _env_int("SAFE_INSIGHT_LLM_MAX_TOKENS", 512)
LLM_TEMPERATURE: Final[float] = _env_float("SAFE_INSIGHT_LLM_TEMPERATURE", 0.2)
LLM_THREADS: Final[int] = _env_int(
    "SAFE_INSIGHT_LLM_THREADS", max(1, (os.cpu_count() or 4) - 1)
)
#: 0 = pure CPU (the assumed 8 GB student laptop). Raise only if a GPU-enabled
#: llama-cpp-python build is installed.
LLM_GPU_LAYERS: Final[int] = _env_int("SAFE_INSIGHT_LLM_GPU_LAYERS", 0)

#: Character budget for the excerpt shown next to each citation in the UI.
CITATION_EXCERPT_CHARS: Final[int] = _env_int("SAFE_INSIGHT_CITATION_EXCERPT_CHARS", 400)

# --------------------------------------------------------------------------- #
# Server / uploads
# --------------------------------------------------------------------------- #
HOST: Final[str] = _env_str("SAFE_INSIGHT_HOST", "127.0.0.1")
PORT: Final[int] = _env_int("SAFE_INSIGHT_PORT", 8765)

#: Only the Tauri dev server and the packaged webview talk to us. Everything
#: else is refused by CORS *and* by the loopback-only bind above.
ALLOWED_ORIGINS: Final[List[str]] = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://tauri.localhost",
]

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES: Final[int] = _env_int("SAFE_INSIGHT_MAX_UPLOAD_MB", 100) * 1024 * 1024

#: When True the socket guard *raises* on any non-loopback connect attempt.
#: When False it only records the attempt. True is the whole point of the project.
STRICT_NETWORK_GUARD: Final[bool] = _env_bool("SAFE_INSIGHT_STRICT_NETWORK", True)


def ensure_directories() -> None:
    """Create every runtime directory. Idempotent; safe to call repeatedly."""
    for directory in (
        DATA_DIR,
        UPLOADS_DIR,
        PROJECTS_DIR,
        INDEX_DIR,
        LOGS_DIR,
        MODELS_DIR,
        EMBEDDING_CACHE_DIR,
        LLM_MODEL_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
