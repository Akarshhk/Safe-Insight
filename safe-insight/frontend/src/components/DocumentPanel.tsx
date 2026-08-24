/**
 * Document management panel: add, list, remove.
 *
 * Uploads run one file at a time and sequentially. Embedding is CPU-bound on the
 * target machine, so firing five uploads in parallel would just thrash the same
 * cores and make every one of them slower - and the per-file progress line here
 * is more useful than a single opaque spinner.
 */

import { useEffect, useRef, useState } from "react";
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

  async function handleFiles(files: FileList | File[] | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    setNotice(null);

    const filesArray = Array.from(files);
    const messages: string[] = [];
    for (let i = 0; i < filesArray.length; i += 1) {
      const file = filesArray[i];
      setProgress(`Indexing ${file.name} (${i + 1}/${filesArray.length})...`);
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

  useEffect(() => {
    const ingestFromNativeClipboard = async (): Promise<boolean> => {
      try {
        const clipboard = await import('tauri-plugin-clipboard-api');
        const { convertFileSrc } = await import('@tauri-apps/api/core');
        
        if (await clipboard.hasFiles()) {
          const osFiles = await clipboard.readFiles();
          if (osFiles && osFiles.length > 0) {
            console.log("[Paste Debug] Found native files:", osFiles);
            const files: File[] = [];
            for (const path of osFiles) {
              const url = convertFileSrc(path);
              const response = await fetch(url);
              const blob = await response.blob();
              const filename = path.split(/[\\/]/).pop() || "unknown";
              files.push(new File([blob], filename));
            }
            if (files.length > 0) {
              await handleFiles(files);
              return true;
            }
          }
        }
        
        if (await clipboard.hasImage()) {
          const imageBase64 = await clipboard.readImageBase64();
          if (imageBase64) {
            console.log("[Paste Debug] Found image from native clipboard");
            const byteCharacters = atob(imageBase64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
              byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {type: 'image/png'});
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const file = new File([blob], `Pasted screenshot - ${timestamp}.png`, { type: 'image/png' });
            await handleFiles([file]);
            return true;
          }
        }
        
        if (await clipboard.hasText()) {
          const text = await clipboard.readText();
          if (text && text.trim()) {
            console.log("[Paste Debug] Found text from native clipboard");
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const file = new File([text], `Pasted text - ${timestamp}.txt`, { type: "text/plain" });
            await handleFiles([file]);
            return true;
          }
        }
      } catch (e) {
        console.error("Tauri native clipboard read failed:", e);
        setError(`Clipboard plugin error: ${String(e)}`);
      }
      return false;
    };

    const handlePaste = async (event: ClipboardEvent) => {
      console.log("[Paste Debug] Paste event fired!", event);
      if (!activeProjectId || busy) return;

      const dt = event.clipboardData;
      if (!dt) return;

      // Try native first! Right-click paste doesn't trigger keydown.
      const handledNatively = await ingestFromNativeClipboard();
      if (handledNatively) {
        event.preventDefault();
        return;
      }
      
      // Fallback to web APIs
      if (dt.files && dt.files.length > 0) {
        const filesArray = Array.from(dt.files);
        const processedFiles = filesArray.map(f => {
          if (f.type.startsWith("image/") && f.name === "image.png") {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            return new File([f], `Pasted screenshot - ${timestamp}.png`, { type: f.type });
          }
          return f;
        });
        
        if (processedFiles.length > 0) {
          await handleFiles(processedFiles);
          return;
        }
      }


      // 3. Fallback to raw text
      const text = dt.getData("text/plain");
      if (text && text.trim()) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const file = new File([text], `Pasted text - ${timestamp}.txt`, { type: "text/plain" });
        await handleFiles([file]);
      }
    };

    const handleKeyDown = async (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') {
        // Don't intercept if user is pasting text into an input or textarea
        const target = e.target as HTMLElement;
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
          return;
        }
        
        console.log("[Paste Debug] Ctrl+V keydown intercepted!");
        if (!activeProjectId || busy) return;
        
        const handled = await ingestFromNativeClipboard();
        if (handled) {
          e.preventDefault();
        }
      }
    };

    document.addEventListener("paste", handlePaste);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("paste", handlePaste);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeProjectId, busy]);

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
