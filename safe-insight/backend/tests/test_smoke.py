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

from app import audit_log, network_guard  # noqa: E402
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
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("huggingface.co", 443)


def test_offline_status_reports_verified():
    network_guard.install_guard()
    status = network_guard.get_status(rerun_self_check=True)
    assert status["guard_installed"] is True
    assert status["outbound_probe_blocked"] is True
