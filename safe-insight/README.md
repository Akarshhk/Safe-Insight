# Safe Insight

**Offline-only document intelligence for people who cannot use cloud AI.**

Safe Insight is a desktop application that lets you ask questions about your own
documents — contracts, patient records, audit files — with a language model that
runs entirely on your machine. No cloud APIs. No telemetry. No network calls of
any kind at runtime. Every answer shows the exact source passages it came from,
and every query is written to a local, tamper-evident audit log.

Built for regulated-industry professionals (legal, healthcare, compliance) whose
documents legally cannot be sent to a third-party service.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [How to demo the offline guarantee](#how-to-demo-the-offline-guarantee)
4. [How it works, in plain language](#how-it-works-in-plain-language) ← *for the project report*
5. [Project layout](#project-layout)
6. [API reference](#api-reference)
7. [Configuration](#configuration)
8. [Engineering decisions and their rationale](#engineering-decisions-and-their-rationale)
9. [Troubleshooting](#troubleshooting)
10. [Known limitations](#known-limitations)
11. [Where to take it next](#where-to-take-it-next)

---

## What it does

| Feature | Status |
|---|---|
| Ingest PDF, DOCX, TXT, MD | ✅ PyMuPDF / python-docx, per-page metadata for PDFs |
| Chunk with overlap, keep provenance | ✅ ~500 tokens, ~50 overlap, sentence-aware |
| Local embeddings | ✅ `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Persistent vector index | ✅ FAISS on disk, incremental add + per-document delete |
| Local generation | ✅ llama-cpp-python, Phi-3.5-mini Q4_K_M GGUF (swappable) |
| Citation trail | ✅ structured citations, clickable to the exact source text |
| Audit log | ✅ SQLite, hash-chained, never transmitted |
| Network kill-switch | ✅ socket-level guard + startup self-check + `/health/offline-check` |
| Desktop shell | ✅ Tauri 2 + React, spawns and supervises the backend |

---

## Quick start

**Prerequisites**

- Python **3.10–3.12** (3.13 wheels for `faiss-cpu` / `llama-cpp-python` are still patchy)
- Node.js 18+
- Rust toolchain (`rustup`) — Tauri needs it
- Windows: **Microsoft Visual C++ Build Tools** and **WebView2** (WebView2 ships with Windows 11)
- ~4 GB free disk (models), 8 GB RAM

### 1. Backend

```bash
cd safe-insight/backend
python -m venv .venv
```

Activate it — Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

### 2. Download the models (the only step that uses the network)

```bash
python download_model.py
```

This fetches ~2.3 GB: the Phi-3.5-mini GGUF and the BGE embedding model, into
`backend/models/`. Useful variants:

```bash
python download_model.py --check
```

```bash
python download_model.py --embedding-only
```

```bash
python download_model.py --llm llama3
```

> `--embedding-only` skips the 2 GB download. The app then runs in
> **retrieval-only mode**: ingestion, search and citations all work and are fully
> demonstrable; the answer box explains that no language model is loaded. This is
> the fastest way to get a working demo while the big download runs.

**After this step you can disconnect from the internet permanently.**

### 3. Start the backend

```bash
python -m app.main
```

It binds `http://127.0.0.1:8765`. Interactive API docs at
`http://127.0.0.1:8765/docs`.

### 4. Frontend

```bash
cd safe-insight/frontend
npm install
```

```bash
npm run tauri dev
```

The Tauri shell **spawns the Python backend itself** (it prefers
`backend/.venv`), so step 3 is optional — but running the backend in its own
terminal is much nicer for debugging, and the shell tolerates an
already-running instance.

### 5. Verify the setup

```bash
cd safe-insight/backend
pytest -q
```

18 tests covering chunking, citation extraction, the audit hash chain and the
network kill-switch. They need no models and finish in under a second.

---

## How to demo the offline guarantee

This is the demo for the viva. It takes about a minute.

1. Start the app normally with Wi-Fi on, and index a document.
2. **Turn on airplane mode / pull the Ethernet cable.**
3. Click the **"Offline Mode: Verified ✅"** badge in the header. It re-runs a
   *live* outbound probe and expands to show the evidence:
   - socket guard installed — yes
   - strict mode — yes (connections raise)
   - outbound probe — blocked
   - escape attempts — 0
   - HTTP client modules loaded — none
4. Ask a question. The full pipeline — embedding, retrieval, generation,
   citations, audit logging — runs exactly as before.
5. Click a citation to reveal the exact source paragraph.
6. Click **Verify audit log** in the footer: the hash chain is re-computed and
   confirmed intact.

To show the guard is real rather than cosmetic, run this while the backend is
running:

```bash
python -c "from app import network_guard as g; g.install_guard(); import socket; socket.socket().connect(('1.1.1.1', 80))"
```

It raises `OutboundNetworkBlocked` instead of connecting.

**What the badge actually proves.** The Python process patches
`socket.connect`, `connect_ex`, `create_connection` and `getaddrinfo` before any
ML library is imported. Every non-loopback destination raises. At startup the
process deliberately *tries* to reach `8.8.8.8:53` and reports whether the
attempt was refused — so the badge reflects a tested behaviour, not a claim in a
comment. The only permitted socket traffic is loopback, which is how the Tauri
window talks to FastAPI.

---

## How it works, in plain language

*(Written to be adapted for a project report; no prior RAG knowledge assumed.)*

A language model on its own has two problems for this use case: it does not know
anything about your private documents, and it cannot tell you where an answer
came from. Retrieval-Augmented Generation (RAG) fixes both by splitting the job
in two — **find the relevant text first, then write an answer using only that
text.**

Safe Insight does this in five stages.

**1. Reading the documents.** When you add a file, the app extracts its text —
page by page for PDFs, so it always knows which page each sentence came from. The
text is then cut into overlapping pieces of roughly 500 tokens (about 375 words),
each overlapping its neighbour by ~50 tokens. Overlap matters: without it, a
sentence that happens to fall on a boundary can be split so that neither half
makes sense on its own. Each piece keeps a label recording its filename, page
number, and position in the document.

**2. Turning text into numbers.** Each piece is passed through an *embedding
model* — a small neural network that converts text into a list of 384 numbers
capturing its meaning. Passages about similar things get similar number lists,
even when they share no words. This model runs on your CPU; nothing is uploaded.

**3. Storing them for fast search.** The number lists go into a FAISS index, a
data structure built for finding "the most similar vectors to this one" quickly.
The index is saved to disk, so restarting the app does not mean re-reading every
document. Adding a new file appends to the existing index rather than rebuilding
it.

**4. Answering a question.** Your question goes through the same embedding
model, producing a comparable list of numbers. FAISS returns the five most
similar passages (configurable). Those passages — and nothing else — are pasted
into a prompt for the local language model, along with an instruction that
amounts to: *answer only from these passages, cite them by number, and if they do
not contain the answer, say so.* The model, a 3.8-billion-parameter Phi-3.5-mini
compressed to about 2 GB so it runs on an ordinary laptop CPU, then writes the
answer.

**5. Proving where the answer came from.** Because the passages were numbered
before being handed to the model, and the model was told to cite those numbers,
every claim in the answer can be traced back. The interface lists each source
underneath the answer with its filename, page number and similarity score;
clicking one reveals the exact original text. Separately, the question, the
answer, the identifiers of every retrieved passage and a timestamp are appended
to a local SQLite database. Each record includes a hash of itself and of the
previous record, so removing or editing an entry later breaks a chain that the
app can re-verify on demand.

**Why "offline" is enforced rather than promised.** Any of the dozens of
libraries involved could, in principle, open a network connection. So instead of
auditing them all, the app replaces Python's own networking functions at startup
with versions that refuse any destination other than the machine itself. It then
tests this by trying to connect out and confirming it was blocked. The green
badge in the interface reports the result of that test.

---

## Project layout

```
safe-insight/
├── README.md                   ← you are here
├── ARCHITECTURE.md             ← data-flow diagrams, design decisions (viva material)
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── download_model.py       ← the ONLY networked script; run once
│   ├── tests/
│   │   └── test_smoke.py       ← 18 model-free tests
│   └── app/
│       ├── __init__.py         ← module map; read this first
│       ├── config.py           ← every tunable; sets offline env vars on import
│       ├── network_guard.py    ← socket kill-switch + self-check
│       ├── ingestion.py        ← file -> text segments -> chunks
│       ├── chunking.py         ← the chunking algorithm
│       ├── embeddings.py       ← local sentence-transformers encoder
│       ├── vector_store.py     ← persistent FAISS index + metadata sidecar
│       ├── llm.py              ← local GGUF generation, grounded prompting
│       ├── citations.py        ← retrieval results -> verifiable citation trail
│       ├── audit_log.py        ← local SQLite compliance log
│       ├── schemas.py          ← HTTP request/response models
│       └── main.py             ← FastAPI app and routes
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    ├── src-tauri/
    │   ├── Cargo.toml
    │   ├── build.rs
    │   ├── tauri.conf.json
    │   ├── capabilities/default.json
    │   └── src/main.rs         ← spawns + supervises the Python backend
    └── src/
        ├── main.tsx
        ├── App.tsx             ← shell, boot sequence, status bar
        ├── api.ts              ← typed client for 127.0.0.1:8765
        ├── types.ts            ← twins of backend/app/schemas.py
        ├── styles.css
        └── components/
            ├── OfflineBadge.tsx
            ├── DocumentPanel.tsx
            ├── ChatPanel.tsx
            └── CitationList.tsx
```

Two files exist that the original brief did not list, both for readability:
`app/schemas.py` (keeps wire formats out of `main.py`) and `app/config.py`
(keeps paths and magic numbers out of everything else).

---

## API reference

All endpoints are on `http://127.0.0.1:8765`. Full interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | index counts, model state, audit stats, offline status |
| `GET` | `/health/offline-check` | offline badge data; `?rerun=true` forces a live probe |
| `POST` | `/documents` | upload + ingest + index one file (multipart) |
| `GET` | `/documents` | list indexed documents |
| `DELETE` | `/documents/{doc_id}` | remove a document and its vectors |
| `GET` | `/documents/{doc_id}/chunks` | every chunk of one document (debugging) |
| `POST` | `/query` | ask a question → answer + citations |
| `GET` | `/chunks/{vector_id}` | untruncated source text behind a citation |
| `GET` | `/audit/queries` | recent audit entries |
| `GET` | `/audit/documents` | ingest/delete history |
| `GET` | `/audit/verify` | re-compute the audit hash chain |
| `POST` | `/audit/export` | dump the audit log to JSON Lines on disk |

Example:

```bash
curl -X POST http://127.0.0.1:8765/query -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period?\",\"top_k\":5}"
```

---

## Configuration

Everything lives in `backend/app/config.py` and every value can be overridden
with an environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `SAFE_INSIGHT_LLM_FILE` | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | GGUF filename to load |
| `SAFE_INSIGHT_LLM_PATH` | `backend/models/llm/<file>` | full path override |
| `SAFE_INSIGHT_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |
| `SAFE_INSIGHT_TOP_K` | `5` | chunks retrieved per query |
| `SAFE_INSIGHT_MIN_SIMILARITY` | `0.20` | cosine floor before a chunk is used |
| `SAFE_INSIGHT_CHUNK_SIZE` | `500` | target chunk size in tokens |
| `SAFE_INSIGHT_CHUNK_OVERLAP` | `50` | overlap in tokens |
| `SAFE_INSIGHT_LLM_CTX` | `4096` | model context window |
| `SAFE_INSIGHT_LLM_THREADS` | `cpu_count - 1` | inference threads |
| `SAFE_INSIGHT_PORT` | `8765` | backend port (also update `frontend/src/api.ts`) |
| `SAFE_INSIGHT_STRICT_NETWORK` | `true` | `false` = log outbound attempts instead of blocking |
| `SAFE_INSIGHT_DATA_DIR` | `backend/data` | where index, uploads and audit log live |

**Swapping the language model** — download the alternative and point at it:

```bash
python download_model.py --llm llama3
```

```bash
set SAFE_INSIGHT_LLM_FILE=Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

No code change is needed: the chat template is read from the GGUF metadata.

> ⚠️ Changing the **embedding** model is different. The FAISS index is built for
> one vector dimension, and vectors from two different models are not comparable.
> Delete `backend/data/index/` and re-ingest after switching. The app detects the
> mismatch and refuses to load rather than returning meaningless scores.

---

## Engineering decisions and their rationale

**SQLite for the audit log, not JSON Lines.** The brief allowed either. SQLite
wins on the four things an audit log is for: it is *queryable* ("every query that
touched `patient_records.pdf` in March" is one indexed `SELECT`, versus a
full-file scan and ad-hoc parsing); it is *crash-durable* (transactional writes,
where a JSONL file killed mid-append leaves a torn final line); it is
*concurrency-safe* (FastAPI serves from a thread pool, and two simultaneous
appends to a text file can interleave); and it is still just one local file with
no server. The one advantage JSONL has — plain text an auditor can read without
tooling — is bought back by `POST /audit/export`, which writes the whole log out
as JSON Lines on demand.

**Hash-chained audit records.** Each row stores the hash of the previous record
plus a hash of itself. Editing or deleting history breaks the chain, and
`GET /audit/verify` reports exactly where. This is not cryptographic
non-repudiation — a determined local user with the source can recompute the chain
— but it makes *silent* tampering impossible, which is what an audit trail is
for.

**Exact FAISS search (`IndexFlatIP`), not an approximate index.** At the target
scale — a few hundred documents, tens of thousands of vectors — a brute-force
scan is sub-millisecond. IVF or HNSW would add a training step, tuning knobs and
recall loss for no user-visible gain. `IndexIDMap2` wraps it to give stable
64-bit ids and, crucially, `remove_ids`, which is what makes per-document delete
possible without a rebuild.

**Vectors normalised at embedding time.** `sentence-transformers` returns unit
vectors, so plain inner product *is* cosine similarity. The vector store stays
simple and the scores shown in the UI are directly interpretable (1.0 identical,
0.0 unrelated).

**Approximate token counting for chunking.** Every real tokenizer wants to
download a vocabulary file on first use, which would break the offline guarantee
at install time. Chunk boundaries only need to be roughly right, so we use a
words-per-token ratio and give the model a generous context margin.

**A similarity floor before generation.** Chunks scoring below 0.20 are dropped
before they reach the prompt. Without this, an off-topic question retrieves the
five least-irrelevant chunks in the corpus and the model dutifully builds an
answer from them. With it, the answer is the honest refusal string.

**Retrieval-only stub mode.** If no GGUF file is present, the app does not crash
— it returns a clearly-labelled placeholder listing the retrieved passages and
their scores. This makes the whole pipeline testable and demonstrable during the
2 GB download, and it keeps a missing model from looking like a broken app.

**Socket patching over dependency auditing.** Any transitive dependency could
open a socket. Patching Python's socket API means we never have to audit the
tree by hand, and the violation log becomes a concrete artefact to point at.

**Duplicate detection by SHA-256.** Re-uploading the same file would otherwise
double-weight its content at retrieval time. The hash is computed before
extraction, so a duplicate costs milliseconds rather than a full re-index.

**Audit entries survive document deletion.** Removing a document takes it out of
future retrieval but leaves the historical record that a query ran against it. An
audit trail you can erase by deleting the evidence is not an audit trail.

---

## Troubleshooting

**`llama-cpp-python` tries to compile from source and fails.**
Prebuilt CPU wheels exist for most platforms. On Windows, install the *Visual
Studio Build Tools* (C++ workload) once and retry, or use the prebuilt index:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**`faiss-cpu` will not install.**
Use Python 3.10–3.12. There are no 3.13 wheels for some platforms yet.

**"No local embedding model found."**
`python download_model.py --embedding-only` has not been run, or
`SAFE_INSIGHT_MODELS_DIR` points somewhere else. Check with
`python download_model.py --check`.

**Backend starts, UI says it cannot reach it.**
Port 8765 is in use. Set `SAFE_INSIGHT_PORT` and update `BACKEND_PORT` in
`frontend/src/api.ts` — they must match.

**Tauri build complains about missing icons.**
`bundle.icon` is intentionally empty; `tauri dev` does not need icons. Before
building an installer, generate them:

```bash
npm run tauri icon path/to/your-icon.png
```

**The offline badge is red and lists a blocked attempt to `huggingface.co`.**
This is the guard working, not failing — but the cause is almost always that the
embedding model was never cached, so `sentence-transformers` tried to download it
and was refused. Run `python download_model.py --embedding-only` and restart the
backend. The badge goes green once nothing in the process attempts an outbound
connection.

**Ingest returns "No extractable text found".**
The PDF is a scan (images, no text layer). Safe Insight bundles no OCR engine —
see *Known limitations*.

**The answer says the documents do not contain enough information, but they do.**
Lower the floor (`SAFE_INSIGHT_MIN_SIMILARITY=0.1`) or raise `top_k`. Very short
chunks or very long questions both push similarity scores down.

---

## Known limitations

These are deliberate v1 scope calls, listed here so they can be discussed rather
than discovered.

1. **No OCR.** Scanned PDFs with no text layer cannot be ingested. Adding
   Tesseract would keep the offline guarantee but adds a large native dependency.
2. **English-centric.** Both embedding models are English. Multilingual documents
   need a multilingual embedding model (and a re-index).
3. **No conversation memory.** Each question is answered independently; "and what
   about clause 4?" will not resolve against the previous turn.
4. **Dev-mode process spawn.** The Tauri shell launches the system Python.
   Shipping a self-contained installer needs PyInstaller — see below.
5. **Page numbers for PDFs only.** DOCX has no page model until it is rendered,
   so those citations use a section index instead.
6. **Single user, no auth.** The backend binds loopback and assumes the machine's
   own login is the security boundary.
7. **Answer quality is bounded by a 3.8B model.** Retrieval and citations are
   solid; the generated prose is what a quantised small model can produce.

---

## Where to take it next

Roughly in order of value per unit of effort:

1. **Streaming answers** — `llama-cpp-python` supports `stream=True`; the UI
   currently waits for the whole response.
2. **Highlight the excerpt inside the source PDF** — `char_start`/`char_end` are
   already stored on every chunk for exactly this.
3. **Hybrid retrieval** — add BM25 keyword search alongside the vector search and
   fuse the rankings. Materially better on names, numbers and clause references.
4. **A reranker** — a small cross-encoder over the top 20 before taking the top 5.
5. **PyInstaller sidecar packaging** — `pyinstaller --onedir app/main.py`, drop
   the result into `src-tauri/binaries/`, and switch `main.rs` to Tauri's sidecar
   API. This turns the project into a genuine double-click installer.
6. **Audit log viewer in the UI** — the endpoints exist (`/audit/queries`,
   `/audit/verify`); only the panel is missing.
7. **Per-document access notes** — who opened what and when, alongside the query
   log.

---

## Licence and models

The application code is yours to license as you wish. The models it downloads
carry their own licences: Phi-3.5-mini (MIT), Llama-3.2-3B (Llama 3.2 Community
Licence), BAAI/bge-small-en-v1.5 (MIT), all-MiniLM-L6-v2 (Apache 2.0). Model
weights are never bundled with this repository.
