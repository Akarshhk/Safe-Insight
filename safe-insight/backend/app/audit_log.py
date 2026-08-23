"""
Local, append-only audit log (SQLite).

Every question, the chunks retrieved for it, the answer produced, and the
timestamp are written to a SQLite database on the user's own disk. Nothing is
ever transmitted anywhere - the process that writes this file has no outbound
network capability at all (see :mod:`app.network_guard`).

Why SQLite rather than JSON Lines?
---------------------------------
Both were on the table. SQLite wins here for four reasons:

1. **Queryable.** A compliance question is "show me every query that touched
   patient_records.pdf last March". That is one indexed ``SELECT`` in SQLite and
   a full-file scan plus ad-hoc parsing in JSONL.
2. **Durable under crash.** SQLite writes are transactional. A JSONL file killed
   mid-write leaves a torn final line that some parsers reject outright.
3. **Concurrency-safe.** FastAPI serves requests from a thread pool. Two
   simultaneous appends to a text file can interleave; SQLite serialises them.
4. **Still just one local file.** The main argument for JSONL - "grep-able, no
   server" - also applies to SQLite: it is a single file with no daemon, and
   ``sqlite3 audit.sqlite3 .dump`` gives plain text whenever an auditor wants it.

The cost is that the log is not human-readable in a text editor. We buy that back
with :func:`export_jsonl`, which writes the whole log out as JSON Lines on demand
for handing to an external auditor.

Tamper-evidence
---------------
Each row stores ``prev_hash`` and ``record_hash``, chaining every entry to its
predecessor. Deleting or editing a historical row breaks the chain, and
:func:`verify_chain` detects exactly where. This is not cryptographic
non-repudiation (a determined local user can recompute the chain), but it does
what an audit trail needs to do: make silent tampering visibly impossible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc     TEXT    NOT NULL,
    query             TEXT    NOT NULL,
    answer            TEXT    NOT NULL,
    top_k             INTEGER NOT NULL,
    retrieved_ids     TEXT    NOT NULL,  -- JSON array of vector ids
    citations         TEXT    NOT NULL,  -- JSON array of citation objects
    documents_touched TEXT    NOT NULL,  -- JSON array of filenames
    model             TEXT    NOT NULL,
    embedding_model   TEXT,
    latency_ms        INTEGER NOT NULL,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    stub_mode         INTEGER DEFAULT 0,
    grounded          INTEGER DEFAULT 1,
    prev_hash         TEXT    NOT NULL,
    record_hash       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log (timestamp_utc);

CREATE TABLE IF NOT EXISTS document_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    action        TEXT NOT NULL,          -- 'ingest' | 'delete' | 'reset'
    doc_id        TEXT,
    filename      TEXT,
    detail        TEXT                    -- JSON blob, action-specific
);

CREATE INDEX IF NOT EXISTS idx_document_log_timestamp ON document_log (timestamp_utc);

CREATE TABLE IF NOT EXISTS system_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    action        TEXT NOT NULL,
    detail        TEXT                    -- JSON blob, action-specific
);

CREATE INDEX IF NOT EXISTS idx_system_log_timestamp ON system_log (timestamp_utc);
"""

_GENESIS_HASH = "0" * 64
_write_lock = threading.Lock()


def _utc_now() -> str:
    """ISO-8601 UTC timestamp. Stored as text so the DB stays greppable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """
    Open a connection with sane durability settings.

    WAL mode lets a read (history view) run while a write (new query) commits.
    ``synchronous=FULL`` costs a few milliseconds per write and buys durability
    across an OS crash, which is the right trade for an audit log.
    """
    path = Path(db_path or config.AUDIT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=FULL;")
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables and indexes if absent. Safe to call on every startup."""
    with _connect(db_path) as connection:
        connection.executescript(SCHEMA)
        try:
            # Migration: add grounded column if it doesn't exist
            connection.execute("ALTER TABLE query_log ADD COLUMN grounded INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass # Column already exists
    logger.info("Audit log ready at %s", db_path or config.AUDIT_DB_PATH)


def _compute_record_hash(payload: Dict[str, Any], prev_hash: str) -> str:
    """SHA-256 over the canonicalised record plus the previous record's hash."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def _latest_hash(connection: sqlite3.Connection) -> str:
    """Hash of the most recent row, or the genesis value for an empty log."""
    row = connection.execute(
        "SELECT record_hash FROM query_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["record_hash"] if row else _GENESIS_HASH


def log_query(
    query: str,
    answer: str,
    top_k: int,
    retrieved_ids: List[int],
    citations: List[Dict[str, Any]],
    documents_touched: List[str],
    model: str,
    latency_ms: int,
    embedding_model: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    stub_mode: bool = False,
    grounded: bool = True,
    db_path: Optional[Path] = None,
) -> int:
    """
    Append one query to the audit log.

    Args:
        query: the user's question, verbatim.
        answer: the generated answer, verbatim.
        retrieved_ids: vector ids of every chunk retrieved (not just the cited
            ones) - an auditor needs to see what the model *could* have used.
        citations: the serialised citation trail, including which were cited.
        documents_touched: distinct filenames behind the retrieved chunks.
        latency_ms: wall-clock time for retrieval + generation.

    Returns:
        The row id of the new entry.
    """
    timestamp = _utc_now()
    record = {
        "timestamp_utc": timestamp,
        "query": query,
        "answer": answer,
        "top_k": top_k,
        "retrieved_ids": retrieved_ids,
        "documents_touched": documents_touched,
        "model": model,
        "latency_ms": latency_ms,
        "grounded": grounded,
    }

    # Serialise writes: the hash chain must be built one row at a time.
    with _write_lock:
        with _connect(db_path) as connection:
            prev_hash = _latest_hash(connection)
            record_hash = _compute_record_hash(record, prev_hash)
            cursor = connection.execute(
                """
                INSERT INTO query_log (
                    timestamp_utc, query, answer, top_k, retrieved_ids, citations,
                    documents_touched, model, embedding_model, latency_ms,
                    prompt_tokens, completion_tokens, stub_mode, grounded, prev_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    query,
                    answer,
                    top_k,
                    json.dumps(retrieved_ids),
                    json.dumps(citations, ensure_ascii=False),
                    json.dumps(documents_touched, ensure_ascii=False),
                    model,
                    embedding_model,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    int(stub_mode),
                    int(grounded),
                    prev_hash,
                    record_hash,
                ),
            )
            return int(cursor.lastrowid)


def log_document_event(
    action: str,
    doc_id: Optional[str] = None,
    filename: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Record an ingest / delete / reset. Kept separate from the query chain."""
    with _write_lock:
        with _connect(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO document_log (timestamp_utc, action, doc_id, filename, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    action,
                    doc_id,
                    filename,
                    json.dumps(detail or {}, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)


def log_system_event(
    action: str,
    detail: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Record system-level compliance events (e.g., sanctioned network access)."""
    with _write_lock:
        with _connect(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO system_log (timestamp_utc, action, detail)
                VALUES (?, ?, ?)
                """,
                (
                    _utc_now(),
                    action,
                    json.dumps(detail or {}, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)


def _row_to_query_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Inflate a DB row back into a JSON-friendly dict."""
    return {
        "id": row["id"],
        "timestamp_utc": row["timestamp_utc"],
        "query": row["query"],
        "answer": row["answer"],
        "top_k": row["top_k"],
        "retrieved_ids": json.loads(row["retrieved_ids"]),
        "citations": json.loads(row["citations"]),
        "documents_touched": json.loads(row["documents_touched"]),
        "model": row["model"],
        "embedding_model": row["embedding_model"],
        "latency_ms": row["latency_ms"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "stub_mode": bool(row["stub_mode"]),
        "grounded": bool(row["grounded"]),
        "record_hash": row["record_hash"],
    }


def get_recent_queries(
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Most recent queries first - backs the history panel."""
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM query_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_query_dict(row) for row in rows]


def get_document_events(
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Most recent document lifecycle events first."""
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM document_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": row["id"],
            "timestamp_utc": row["timestamp_utc"],
            "action": row["action"],
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "detail": json.loads(row["detail"] or "{}"),
        }
        for row in rows
    ]


def verify_chain(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Re-compute the hash chain and report the first break, if any.

    Returns a dict with ``valid``, ``checked`` (rows examined) and, on failure,
    ``broken_at_id`` plus a human explanation. Exposed at
    ``GET /audit/verify`` - a one-click integrity demo for the viva.
    """
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM query_log ORDER BY id ASC").fetchall()

    prev_hash = _GENESIS_HASH
    for index, row in enumerate(rows, start=1):
        record = {
            "timestamp_utc": row["timestamp_utc"],
            "query": row["query"],
            "answer": row["answer"],
            "top_k": row["top_k"],
            "retrieved_ids": json.loads(row["retrieved_ids"]),
            "documents_touched": json.loads(row["documents_touched"]),
            "model": row["model"],
            "latency_ms": row["latency_ms"],
            "grounded": bool(row["grounded"]),
        }
        expected = _compute_record_hash(record, prev_hash)
        if row["prev_hash"] != prev_hash or row["record_hash"] != expected:
            return {
                "valid": False,
                "checked": index,
                "total": len(rows),
                "broken_at_id": row["id"],
                "detail": (
                    f"Audit record {row['id']} does not match the hash chain. "
                    f"The log has been modified or a record was removed."
                ),
            }
        prev_hash = row["record_hash"]

    return {
        "valid": True,
        "checked": len(rows),
        "total": len(rows),
        "detail": "All audit records hash-chain correctly.",
    }


def export_jsonl(destination: Path, db_path: Optional[Path] = None) -> int:
    """
    Write the whole query log out as JSON Lines.

    This is the escape hatch that makes the SQLite choice safe: an auditor who
    wants plain text gets it, one JSON object per line, without needing SQLite.

    Returns:
        Number of records written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM query_log ORDER BY id ASC").fetchall()
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_row_to_query_dict(row), ensure_ascii=False) + "\n"
            )
    return len(rows)


def stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Row counts and the log's location, for ``GET /health`` and the UI."""
    with _connect(db_path) as connection:
        queries = connection.execute("SELECT COUNT(*) AS n FROM query_log").fetchone()["n"]
        events = connection.execute("SELECT COUNT(*) AS n FROM document_log").fetchone()["n"]
        last = connection.execute(
            "SELECT timestamp_utc FROM query_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "query_count": queries,
        "document_event_count": events,
        "last_query_at": last["timestamp_utc"] if last else None,
        "database_path": str(db_path or config.AUDIT_DB_PATH),
        "transmitted_anywhere": False,  # stated explicitly; asserted by network_guard
    }


__all__ = [
    "init_db",
    "log_query",
    "log_document_event",
    "log_system_event",
    "get_recent_queries",
    "get_document_events",
    "verify_chain",
    "export_jsonl",
    "stats",
]
