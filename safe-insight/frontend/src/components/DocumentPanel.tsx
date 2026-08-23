/**
 * Document management panel: add, list, remove.
 *
 * Uploads run one file at a time and sequentially. Embedding is CPU-bound on the
 * target machine, so firing five uploads in parallel would just thrash the same
 * cores and make every one of them slower - and the per-file progress line here
 * is more useful than a single opaque spinner.
 */

import { useRef, useState } from "react";
import { deleteDocument, uploadDocument } from "../api";
import type { DocumentInfo } from "../types";

interface Props {
  activeProjectId: string | null;
  documents: DocumentInfo[];
  loading: boolean;
  /** Called after any change so the parent can re-fetch the list. */
  onChanged: () => void | Promise<void>;
}

const ACCEPTED = ".pdf,.docx,.txt,.md";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentPanel({ activeProjectId, documents, loading, onChanged }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    setNotice(null);

    const messages: string[] = [];
    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      setProgress(`Indexing ${file.name} (${i + 1}/${files.length})...`);
      try {
        if (!activeProjectId) throw new Error("No project selected");
        const result = await uploadDocument(activeProjectId, file);
        messages.push(
          result.duplicate_of
            ? `${result.filename}: already indexed, skipped`
            : `${result.filename}: ${result.chunk_count} chunks in ${result.elapsed_ms} ms`,
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        break;
      }
    }

    setProgress(null);
    setBusy(false);
    if (messages.length > 0) setNotice(messages.join(" · "));
    if (fileInput.current) fileInput.current.value = "";
    await onChanged();
  }

  async function handleDelete(doc: DocumentInfo) {
    // Deleting removes the document from future retrieval only; audit history
    // that references it is deliberately preserved.
    if (!window.confirm(`Remove "${doc.filename}" from the index?`)) return;
    setBusy(true);
    setError(null);
    try {
      if (!activeProjectId) throw new Error("No project selected");
      await deleteDocument(activeProjectId, doc.doc_id);
      setNotice(`Removed ${doc.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      await onChanged();
    }
  }

  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunk_count, 0);

  return (
    <section className="panel documents">
      <header className="panel__header">
        <h2>Documents</h2>
        <span className="muted small">
          {documents.length} file{documents.length === 1 ? "" : "s"} · {totalChunks} chunks
        </span>
      </header>

      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          void handleFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPTED}
          multiple
          disabled={busy || !activeProjectId}
          onChange={(event) => void handleFiles(event.target.files)}
        />
        <p className="muted small">Drag files here, or use the picker. PDF, DOCX, TXT, MD.</p>
      </div>

      {progress && <p className="status status--busy">{progress}</p>}
      {notice && !progress && <p className="status status--ok">{notice}</p>}
      {error && <p className="status status--error">{error}</p>}

      <ul className="doc-list">
        {loading && documents.length === 0 && <li className="muted">Loading...</li>}
        {!loading && documents.length === 0 && (
          <li className="muted">
            No documents indexed yet. Add one to start asking questions.
          </li>
        )}
        {documents.map((doc) => (
          <li key={doc.doc_id} className="doc-list__item">
            <div className="doc-list__main">
              <span className="doc-list__name" title={doc.filename}>
                {doc.filename}
              </span>
              <span className="muted small">
                {doc.file_type.toUpperCase()} · {formatBytes(doc.size_bytes)} ·{" "}
                {doc.page_count ? `${doc.page_count} pages · ` : ""}
                {doc.chunk_count} chunks
              </span>
            </div>
            <button
              type="button"
              className="btn btn--danger btn--small"
              disabled={busy}
              onClick={() => void handleDelete(doc)}
              aria-label={`Remove ${doc.filename}`}
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
