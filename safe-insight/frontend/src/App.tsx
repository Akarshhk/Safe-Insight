/**
 * Application shell.
 *
 * Owns three things and delegates everything else:
 *   1. backend readiness (the Python process starts after the window does),
 *   2. the document list, shared between the document panel and the chat panel,
 *   3. the status bar (offline badge, model state, audit counters).
 */

import { useCallback, useEffect, useState } from "react";
import { getProject, verifyAudit, waitForBackend, getSetupStatus } from "./api";
import ChatPanel from "./components/ChatPanel";
import DocumentPanel from "./components/DocumentPanel";
import OfflineBadge from "./components/OfflineBadge";
import ProjectSidebar from "./components/ProjectSidebar";
import SetupWizard from "./components/SetupWizard";
import type { DocumentInfo, HealthResponse, ChatMessage } from "./types";

type BootState = "starting" | "ready" | "failed";

export default function App() {
  const [boot, setBoot] = useState<BootState>("starting");
  const [bootError, setBootError] = useState<string | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [auditNotice, setAuditNotice] = useState<string | null>(null);

  const refreshProject = useCallback(async () => {
    if (!activeProjectId) {
      setDocuments([]);
      setMessages([]);
      return;
    }
    setDocsLoading(true);
    try {
      const proj = await getProject(activeProjectId);
      setDocuments(proj.documents || []);
      
      const formattedMessages: ChatMessage[] = (proj.messages || []).map((m: any) => ({
        id: String(m.id),
        role: m.role,
        text: m.content,
        citations: m.citations,
      }));
      setMessages(formattedMessages);
    } catch (err) {
      console.error(err);
    } finally {
      setDocsLoading(false);
    }
  }, [activeProjectId]);

  useEffect(() => {
    refreshProject();
  }, [refreshProject]);

  // Wait for the Python process the Tauri shell spawned, then load state.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await waitForBackend();
        if (cancelled) return;
        setHealth(status);
        
        const setup = await getSetupStatus();
        if (cancelled) return;
        
        if (!setup.model_present) {
          setShowSetup(true);
        } else {
          setBoot("ready");
        }
      } catch (err) {
        if (cancelled) return;
        setBootError(err instanceof Error ? err.message : String(err));
        setBoot("failed");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleVerifyAudit() {
    try {
      const result = await verifyAudit();
      setAuditNotice(
        result.valid
          ? `Audit log intact: ${result.checked} record(s) hash-chain correctly.`
          : `Audit log FAILED verification at record ${result.broken_at_id}.`,
      );
    } catch (err) {
      setAuditNotice(err instanceof Error ? err.message : String(err));
    }
  }

  if (boot !== "ready") {
    if (showSetup) {
      return (
        <SetupWizard onComplete={() => {
          setShowSetup(false);
          setBoot("ready");
        }} />
      );
    }

    return (
      <div className="boot">
        <h1>Safe Insight</h1>
        {boot === "starting" ? (
          <>
            <p>Starting the local engine...</p>
            <p className="muted small">
              Loading the embedding model and language model into memory. First
              launch takes longest.
            </p>
          </>
        ) : (
          <>
            <p className="status status--error">Could not reach the backend.</p>
            <p className="muted small">{bootError}</p>
            <p className="muted small">
              Start it manually with <code>python -m app.main</code> from the{" "}
              <code>backend/</code> directory and reload this window.
            </p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>Safe Insight</h1>
          <p className="muted small">
            Offline document intelligence · every answer cited · every query logged locally
          </p>
        </div>
        <OfflineBadge />
      </header>

      <main className="app__body">
        <ProjectSidebar 
          activeProjectId={activeProjectId} 
          onProjectSelect={setActiveProjectId} 
        />
        
        <div className="app__content">
          <DocumentPanel
            activeProjectId={activeProjectId}
            documents={documents}
            loading={docsLoading}
            onChanged={refreshProject}
          />
          <ChatPanel
            activeProjectId={activeProjectId}
            hasDocuments={documents.length > 0}
            topK={topK}
            onTopKChange={setTopK}
            initialMessages={messages}
          />
        </div>
      </main>

      <footer className="app__footer">
        <span className="muted small">
          {health?.llm.model_present
            ? `Model: ${health.llm.model_file}`
            : "No language model - retrieval-only mode"}
          {health?.index.embedding_model ? ` · Embeddings: ${health.index.embedding_model}` : ""}
          {` · ${health?.index.vectors ?? 0} vectors indexed`}
          {` · ${health?.audit.query_count ?? 0} queries logged`}
        </span>
        <button type="button" className="btn btn--small" onClick={() => void handleVerifyAudit()}>
          Verify audit log
        </button>
        {auditNotice && <span className="muted small">{auditNotice}</span>}
      </footer>
    </div>
  );
}
