/**
 * Chat interface.
 *
 * Deliberately stateless across restarts: the transcript lives in component
 * state only. The durable record of every question and answer is the backend's
 * audit log, which is the artefact that actually matters for compliance - a
 * second, editable copy in the UI would only muddy that story.
 */

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { askQuestionStream, ApiError } from "../api";
import type { ChatMessage } from "../types";
import CitationList from "./CitationList";

interface Props {
  activeProjectId: string | null;
  /** Disables input when nothing is indexed yet. */
  hasDocuments: boolean;
  topK: number;
  onTopKChange: (value: number) => void;
  initialMessages: ChatMessage[];
}

let messageCounter = 0;
const nextId = () => `m${(messageCounter += 1)}`;

/** Same wording the backend saves, so the chat reads the same after a reload. */
const STOPPED_NOTE = "[Stopped before the answer was finished.]";

export default function ChatPanel({ activeProjectId, hasDocuments, topK, onTopKChange, initialMessages }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [strictMode, setStrictMode] = useState(false);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  /** The answer being written right now, if any - what the Stop button aborts. */
  const answering = useRef<AbortController | null>(null);
  const currentProject = useRef(activeProjectId);

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  // Switching project (or closing the panel) stops the answer in progress, so it
  // can't land in another project's chat; the backend stops generating as well.
  useEffect(() => {
    currentProject.current = activeProjectId;
    return () => answering.current?.abort();
  }, [activeProjectId]);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || pending) return;

    const projectId = activeProjectId;
    const asked: ChatMessage = { id: nextId(), role: "user", text: question };
    const answerId = nextId();
    setMessages((current) => [
      ...current,
      asked,
      { id: answerId, role: "assistant", text: "", streaming: true },
    ]);
    setInput("");
    setPending(true);

    const controller = new AbortController();
    answering.current = controller;
    let written = "";
    // Update the answer bubble in place. If the transcript was reloaded meanwhile
    // (e.g. a document upload refreshed it), put this exchange back rather than
    // lose it; if the user has moved to another project, leave that chat alone.
    const showAnswer = (patch: Partial<ChatMessage>) => {
      if (currentProject.current !== projectId) return;
      setMessages((current) => {
        if (current.some((m) => m.id === answerId)) {
          return current.map((m) => (m.id === answerId ? { ...m, ...patch } : m));
        }
        const question = current.some((m) => m.id === asked.id) ? [] : [asked];
        return [...current, ...question, { id: answerId, role: "assistant", text: "", ...patch }];
      });
    };

    try {
      if (!projectId) throw new Error("No active project");
      const response = await askQuestionStream(
        projectId,
        question,
        topK,
        strictMode,
        (text) => {
          written += text;
          showAnswer({ text: written });
        },
        controller.signal,
      );
      showAnswer({
        text: response.answer,
        streaming: false,
        citations: response.citations,
        meta: {
          latencyMs: response.latency_ms,
          model: response.model,
          retrievedCount: response.retrieved_count,
          stubMode: response.stub_mode,
          auditId: response.audit_id,
          grounded: response.grounded,
        },
      });
    } catch (err) {
      if (controller.signal.aborted && !(err instanceof ApiError)) {
        // Stopped on purpose: keep what was written, marked like the saved copy.
        showAnswer({ text: `${written.trim()}\n\n${STOPPED_NOTE}`.trim(), streaming: false });
      } else {
        const reason =
          err instanceof ApiError && err.status === 503
            ? `${err.message}\n\nThe embedding model is required before anything can be searched.`
            : err instanceof Error
              ? err.message
              : String(err);
        // Keep the partial answer visible above the reason it ended.
        showAnswer({
          text: written.trim() ? `${written.trim()}\n\n${reason}` : reason,
          streaming: false,
          isError: true,
        });
      }
    } finally {
      if (answering.current === controller) answering.current = null;
      setPending(false);
      // Wait for pending state to update the DOM, then refocus
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  }

  return (
    <section className="panel chat">
      <header className="panel__header">
        <h2>Ask your documents</h2>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <label className="topk" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={strictMode}
              onChange={(e) => setStrictMode(e.target.checked)}
            />
            <span className="muted small">Strict document mode</span>
          </label>
          <label className="topk">
            <span className="muted small">Sources per answer</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(event) => onTopKChange(Number(event.target.value) || 1)}
            />
            <span className="muted" style={{ fontSize: '0.75rem', marginTop: '0.25rem', display: 'block' }}>(fewer sources = faster answers)</span>
          </label>
        </div>
      </header>

      <div className="transcript">
        {messages.length === 0 && (
          <div className="empty">
            <p>
              {!activeProjectId 
                ? "Select or create a project to start."
                : "Ask a question about your documents."}
            </p>
            {activeProjectId && !hasDocuments && (
              <p className="muted" style={{ marginBottom: '1rem' }}>
                💡 Add a document on the left to get started.
              </p>
            )}
            <p className="muted small">
              Answers are generated on this machine and every one comes with the
              exact source passages it was built from.
            </p>
          </div>
        )}

        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="turn turn--user">
              <div className="bubble bubble--user">{message.text}</div>
            </div>
          ) : (
            <div key={message.id} className="turn turn--assistant">
              <div className={`bubble bubble--assistant ${message.isError ? "bubble--error" : ""}`}>
                {message.streaming && !message.text ? (
                  <p className="muted">Reading your question and documents...</p>
                ) : (
                  <p className={`answer ${message.streaming ? "answer--streaming" : ""}`}>
                    {message.text}
                  </p>
                )}

                {message.meta && (
                  <p className="muted small answer-meta">
                    {message.meta.grounded === false && (
                      <span style={{ color: 'var(--accent-color)', fontWeight: 500 }}>
                        General knowledge — not from your documents
                      </span>
                    )}
                    {message.meta.grounded !== false && (
                      <>
                        {message.meta.retrievedCount} passage
                        {message.meta.retrievedCount === 1 ? "" : "s"} retrieved
                      </>
                    )}
                    {" · "}
                    {message.meta.latencyMs} ms · {message.meta.model} · audit #
                    {message.meta.auditId}
                    {message.meta.stubMode && " · retrieval-only (no LLM loaded)"}
                  </p>
                )}

                {activeProjectId && message.citations && message.citations.length > 0 && (
                  <CitationList projectId={activeProjectId} citations={message.citations} />
                )}
              </div>
            </div>
          ),
        )}

        <div ref={transcriptEnd} />
      </div>

      <form className="composer" onSubmit={(event) => void submit(event)}>
        <textarea
          ref={inputRef}
          value={input}
          rows={2}
          placeholder="Ask a question about your documents..."
          disabled={pending || !activeProjectId}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter makes a new line.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
        />
        {pending ? (
          // Always a way out: stops the answer here and on the backend.
          <button type="button" className="btn" onClick={() => answering.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn btn--primary" disabled={!input.trim() || !activeProjectId}>
            Ask
          </button>
        )}
      </form>
    </section>
  );
}
