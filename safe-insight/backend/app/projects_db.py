import sqlite3
import json
import uuid
import threading
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional
from pathlib import Path

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    added_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT NOT NULL,  -- JSON
    created_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
"""

_write_lock = threading.Lock()

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = config.APP_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    # Enable foreign keys
    connection.execute("PRAGMA foreign_keys = ON;")
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=FULL;")
        yield connection
        connection.commit()
    finally:
        connection.close()

def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)

def create_project(name: str) -> Dict[str, Any]:
    project_id = str(uuid.uuid4())
    created_at = _utc_now()
    
    # Create the directory on disk
    (config.PROJECTS_DIR / project_id).mkdir(parents=True, exist_ok=True)
    
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (project_id, name, created_at)
        )
    return {"id": project_id, "name": name, "created_at": created_at}

def list_projects() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT p.id, p.name, p.created_at, COUNT(d.id) as doc_count 
            FROM projects p
            LEFT JOIN documents d ON p.id = d.project_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]

def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        proj = conn.execute("SELECT id, name, created_at FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not proj:
            return None
        
        docs = conn.execute("SELECT id, filename, added_at FROM documents WHERE project_id = ? ORDER BY added_at DESC", (project_id,)).fetchall()
        messages = conn.execute("SELECT id, role, content, citations, created_at FROM messages WHERE project_id = ? ORDER BY id ASC", (project_id,)).fetchall()
        
    return {
        "id": proj["id"],
        "name": proj["name"],
        "created_at": proj["created_at"],
        "documents": [dict(d) for d in docs],
        "messages": [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "citations": json.loads(m["citations"]),
                "created_at": m["created_at"],
            }
            for m in messages
        ]
    }

def delete_project(project_id: str) -> None:
    with _write_lock, _connect() as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

def add_document(project_id: str, doc_id: str, filename: str) -> None:
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, project_id, filename, added_at) VALUES (?, ?, ?, ?)",
            (doc_id, project_id, filename, _utc_now())
        )

def delete_document(doc_id: str) -> None:
    with _write_lock, _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

def add_message(project_id: str, role: str, content: str, citations: List[Dict[str, Any]] = None) -> None:
    if citations is None:
        citations = []
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO messages (project_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, role, content, json.dumps(citations, ensure_ascii=False), _utc_now())
        )
