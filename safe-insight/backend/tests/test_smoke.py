"""
Smoke tests for the parts of Safe Insight that need no downloaded models.

These cover chunking, citation formatting, the audit log (including its hash
chain), and the network kill-switch. Together they exercise the logic a marker
or examiner is most likely to poke at, and they run in about a second.

    cd backend
    pytest -q

Tests that need FAISS, torch or a GGUF file are intentionally absent: they would
turn a one-second check into a multi-gigabyte prerequisite.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

# Allow `pytest` to be run from the backend/ directory without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audit_log, config, network_guard, projects_db, references  # noqa: E402
from app.chunking import (  # noqa: E402
    TextSegment,
    chunk_document,
    estimate_tokens,
    normalise_whitespace,
)
from app.citations import (  # noqa: E402
    extract_cited_indices,
    format_location,
    truncate_excerpt,
)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunking_respects_size_and_keeps_pages():
    """Chunks stay near the target size and inherit their page number."""
    sentence = "The tenant shall pay rent on the first day of each month. "
    segments = [
        TextSegment(text=sentence * 120, page_number=1),
        TextSegment(text=sentence * 60, page_number=2),
    ]
    chunks = chunk_document(segments, chunk_size_tokens=200, overlap_tokens=20)

    assert len(chunks) > 1
    # chunk_index is document-scoped, sequential, and gap-free.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # No chunk blows well past the budget.
    assert all(c.approx_tokens <= 260 for c in chunks)
    # Every chunk knows which page it came from.
    assert {c.page_number for c in chunks} == {1, 2}


def test_chunking_handles_text_with_no_punctuation():
    """A single enormous 'sentence' is hard-split rather than dropped."""
    chunks = chunk_document(
        [TextSegment(text="word " * 2000, page_number=None)],
        chunk_size_tokens=100,
        overlap_tokens=0,
    )
    assert len(chunks) > 5
    assert all(c.text.strip() for c in chunks)


def test_empty_segment_yields_no_chunks():
    assert chunk_document([TextSegment(text="   \n\n  ", page_number=1)]) == []


def test_normalise_whitespace_collapses_pdf_spacing():
    assert normalise_whitespace("a    b\t\tc \n\n\n\n d") == "a b c\n\nd"


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Rent is due monthly [1].", [1]),
        ("Both clauses apply [1, 3].", [1, 3]),
        ("See [2][4] for detail.", [2, 4]),
        ("No citation here.", []),
        ("Repeated [1] and again [1].", [1]),
    ],
)
def test_extract_cited_indices(answer, expected):
    assert extract_cited_indices(answer) == expected


def test_format_location_prefers_page_number():
    assert format_location("lease.pdf", 4, 11) == "lease.pdf, page 4"
    assert format_location("notes.docx", None, 2) == "notes.docx, section 3"


def test_truncate_excerpt_marks_truncation_and_breaks_on_words():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    excerpt, truncated = truncate_excerpt(text, max_chars=20)
    assert truncated is True
    assert excerpt.endswith("...")
    assert "gam" not in excerpt or "gamma" in excerpt   # never a half word

    short, truncated = truncate_excerpt("short text", max_chars=100)
    assert (short, truncated) == ("short text", False)


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def test_audit_log_roundtrip_and_chain(tmp_path: Path):
    """Entries persist, read back newest-first, and hash-chain correctly."""
    db = tmp_path / "audit.sqlite3"
    audit_log.init_db(db)

    for n in range(3):
        audit_log.log_query(
            query=f"question {n}",
            answer=f"answer {n} [1]",
            top_k=5,
            retrieved_ids=[n * 10, n * 10 + 1],
            citations=[{"index": 1, "vector_id": n * 10, "filename": "a.pdf"}],
            documents_touched=["a.pdf"],
            model="test-model",
            latency_ms=42,
            db_path=db,
        )

    entries = audit_log.get_recent_queries(db_path=db)
    assert len(entries) == 3
    assert entries[0]["query"] == "question 2"          # newest first
    assert entries[0]["retrieved_ids"] == [20, 21]

    assert audit_log.verify_chain(db_path=db)["valid"] is True
    assert audit_log.stats(db_path=db)["query_count"] == 3


def test_audit_chain_detects_tampering(tmp_path: Path):
    """Editing a stored answer breaks the chain and the break is located."""
    import sqlite3

    db = tmp_path / "audit.sqlite3"
    audit_log.init_db(db)
    for n in range(3):
        audit_log.log_query(
            query=f"q{n}",
            answer=f"a{n}",
            top_k=5,
            retrieved_ids=[n],
            citations=[],
            documents_touched=[],
            model="m",
            latency_ms=1,
            db_path=db,
        )

    connection = sqlite3.connect(db)
    connection.execute("UPDATE query_log SET answer = 'tampered' WHERE id = 2")
    connection.commit()
    connection.close()

    result = audit_log.verify_chain(db_path=db)
    assert result["valid"] is False
    assert result["broken_at_id"] == 2


def test_audit_export_jsonl(tmp_path: Path):
    db = tmp_path / "audit.sqlite3"
    audit_log.init_db(db)
    audit_log.log_query(
        query="q", answer="a", top_k=1, retrieved_ids=[1], citations=[],
        documents_touched=["d.pdf"], model="m", latency_ms=1, db_path=db,
    )
    destination = tmp_path / "export.jsonl"
    assert audit_log.export_jsonl(destination, db_path=db) == 1
    assert '"query": "q"' in destination.read_text(encoding="utf-8")


def test_audit_chain_covers_citations(tmp_path: Path):
    """Editing a stored citation (not just the answer) breaks the chain too."""
    import sqlite3

    db = tmp_path / "audit.sqlite3"
    audit_log.init_db(db)
    audit_log.log_query(
        query="q", answer="a [1]", top_k=1, retrieved_ids=[7],
        citations=[{"index": 1, "vector_id": 7, "score": 0.81}],
        documents_touched=["a.pdf"], model="m", latency_ms=1, db_path=db,
    )
    assert audit_log.verify_chain(db_path=db)["valid"] is True

    connection = sqlite3.connect(db)
    connection.execute("UPDATE query_log SET citations = '[]' WHERE id = 1")
    connection.commit()
    connection.close()

    result = audit_log.verify_chain(db_path=db)
    assert result["valid"] is False
    assert result["broken_at_id"] == 1


def test_audit_log_from_an_older_build_still_verifies(tmp_path: Path):
    """Rows hashed by older builds (fewer fields) verify alongside new ones."""
    import sqlite3

    db = tmp_path / "audit.sqlite3"
    old_record = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00", "query": "q0", "answer": "a0",
        "top_k": 5, "retrieved_ids": [1], "documents_touched": ["a.pdf"],
        "model": "m", "latency_ms": 1,
    }
    # Oldest builds did not hash `grounded`; the next ones did.
    first_hash = audit_log._compute_record_hash(old_record, "0" * 64)
    second_hash = audit_log._compute_record_hash({**old_record, "grounded": True}, first_hash)

    connection = sqlite3.connect(db)
    connection.execute(  # query_log as older builds created it
        """
        CREATE TABLE query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT NOT NULL,
            query TEXT NOT NULL, answer TEXT NOT NULL, top_k INTEGER NOT NULL,
            retrieved_ids TEXT NOT NULL, citations TEXT NOT NULL,
            documents_touched TEXT NOT NULL, model TEXT NOT NULL, embedding_model TEXT,
            latency_ms INTEGER NOT NULL, prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0, stub_mode INTEGER DEFAULT 0,
            grounded INTEGER DEFAULT 1, prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL
        )
        """
    )
    for prev_hash, record_hash in (("0" * 64, first_hash), (first_hash, second_hash)):
        connection.execute(
            "INSERT INTO query_log (timestamp_utc, query, answer, top_k, retrieved_ids, "
            "citations, documents_touched, model, latency_ms, grounded, prev_hash, record_hash) "
            "VALUES (?, 'q0', 'a0', 5, '[1]', '[]', '[\"a.pdf\"]', 'm', 1, 1, ?, ?)",
            (old_record["timestamp_utc"], prev_hash, record_hash),
        )
    connection.commit()
    connection.close()

    audit_log.init_db(db)                    # migrates the old table in place
    audit_log.log_query(
        query="q1", answer="a1", top_k=5, retrieved_ids=[2], citations=[],
        documents_touched=[], model="m", latency_ms=1, project_id="p1", db_path=db,
    )

    assert audit_log.verify_chain(db_path=db)["valid"] is True
    assert audit_log.get_recent_queries(db_path=db)[0]["project_id"] == "p1"


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
def test_unknown_project_is_404_and_touches_no_files(tmp_path: Path, monkeypatch):
    """Project ids become directory names, so only known ids may get that far."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.sqlite3")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    # If the 404 ever regresses, a real query must not land in the real audit log.
    monkeypatch.setattr(config, "AUDIT_DB_PATH", tmp_path / "audit.sqlite3")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    projects_db.init_db()

    client = TestClient(app)        # no `with`: skips model loading at startup
    assert client.post("/projects/nope/query", json={"question": "q"}).status_code == 404
    assert client.delete("/projects/nope").status_code == 404
    assert not (tmp_path / "projects").exists()


def test_readding_a_document_to_its_project_is_a_no_op(tmp_path: Path, monkeypatch):
    """A duplicate upload re-registers the existing doc_id; that must not fail."""
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.sqlite3")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    projects_db.init_db()
    project = projects_db.create_project("p")

    projects_db.add_document(project["id"], "doc-1", "a.pdf")
    projects_db.add_document(project["id"], "doc-1", "a.pdf")   # must NOT raise

    assert len(projects_db.get_project(project["id"])["documents"]) == 1


# --------------------------------------------------------------------------- #
# Questions that name or point at a document
# --------------------------------------------------------------------------- #
def _doc(filename: str, ingested_at: str):
    from types import SimpleNamespace

    return SimpleNamespace(filename=filename, ingested_at=ingested_at)


PROJECT_DOCS = [
    _doc("Akarsh_Kalapurmut_DSATM.pdf", "2026-10-08T10:05:00"),     # newest upload
    _doc("Perceptron Convergence Theorem.pdf", "2026-10-08T10:01:00"),
    _doc("BI module 1 ppt.pdf", "2026-10-08T10:00:00"),
    _doc("Question_bank_DL_till_IA1.pdf", "2026-10-08T09:59:00"),
]


@pytest.mark.parametrize(
    "question, expected",
    [
        # by name: the full filename, or two of its words
        ("what do you see in akarsh_kalapurmut_dsatm.pdf", ["Akarsh_Kalapurmut_DSATM.pdf"]),
        ("summarise the perceptron convergence notes", ["Perceptron Convergence Theorem.pdf"]),
        ("what is in the BI module", ["BI module 1 ppt.pdf"]),
        # by pointing: the newest upload
        ("what is this doc about", ["Akarsh_Kalapurmut_DSATM.pdf"]),
        ("ive attached here", ["Akarsh_Kalapurmut_DSATM.pdf"]),
        ("tell me about my resume", ["Akarsh_Kalapurmut_DSATM.pdf"]),
        # neither: ordinary search / general chat
        ("how is the weather today", []),
        ("what is a perceptron", []),
        ("what is in module 3", []),
    ],
)
def test_referred_documents(question, expected):
    found = references.referred_documents(question, PROJECT_DOCS)
    assert [doc.filename for doc in found] == expected


def test_single_word_filename_matches_that_word():
    assert references.names_document("what does my resume say", "resume.pdf")
    assert not references.names_document("clause 1 please", "1.pdf")


# --------------------------------------------------------------------------- #
# Network kill-switch
# --------------------------------------------------------------------------- #
def test_guard_blocks_outbound_but_allows_loopback():
    """The core offline guarantee, asserted directly."""
    network_guard.install_guard()

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(network_guard.OutboundNetworkBlocked):
        probe.connect(("8.8.8.8", 53))
    probe.close()

    # Loopback must still work - it is how Tauri talks to FastAPI.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(2.0)
    client.connect(("127.0.0.1", port))     # must NOT raise
    client.close()
    server.close()


def test_guard_blocks_remote_dns():
    network_guard.install_guard()
    # A gaierror, so HTTP libraries handle it as a normal resolution failure...
    with pytest.raises(socket.gaierror) as excinfo:
        socket.getaddrinfo("huggingface.co", 443)
    # ...and still the guard's own exception for code that checks for that.
    assert isinstance(excinfo.value, network_guard.OutboundNetworkBlocked)


def test_offline_status_reports_verified(monkeypatch):
    network_guard.install_guard()
    # Other tests deliberately trip the guard; start from a clean record.
    monkeypatch.setattr(network_guard, "_violations", [])
    status = network_guard.get_status(rerun_self_check=True)
    assert status["guard_installed"] is True
    assert status["outbound_probe_blocked"] is True
    # The self-check's own probe is not an escape, so the badge can go green.
    assert status["violation_count"] == 0
    assert status["offline_verified"] is True

    # ...while a real attempt from elsewhere still counts.
    with socket.socket() as probe, pytest.raises(network_guard.OutboundNetworkBlocked):
        probe.connect(("1.1.1.1", 80))
    assert network_guard.get_status()["offline_verified"] is False
