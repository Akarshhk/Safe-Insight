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
import { askQuestion, ApiError } from "../api";
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

export default function ChatPanel({ activeProjectId, hasDocuments, topK, onTopKChange, initialMessages }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [strictMode, setStrictMode] = useState(false);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || pending) return;

    setMessages((current) => [...current, { id: nextId(), role: "user", text: question }]);
    setInput("");
    setPending(true);

    try {
      if (!activeProjectId) throw new Error("No active project");
      const response = await askQuestion(activeProjectId, question, topK, strictMode);
      setMessages((current) => [
        ...current,
        {
          id: nextId(),
          role: "assistant",
          text: response.answer,
          citations: response.citations,
          meta: {
            latencyMs: response.latency_ms,
            model: response.model,
            retrievedCount: response.retrieved_count,
            stubMode: response.stub_mode,
            auditId: response.audit_id,
            grounded: response.grounded,
          },
        },
      ]);
    } catch (err) {
      const text =
        err instanceof ApiError && err.status === 503
          ? `${err.message}\n\nThe embedding model is required before anything can be searched.`
          : err instanceof Error
            ? err.message
            : String(err);
      setMessages((current) => [
        ...current,
        { id: nextId(), role: "assistant", text, isError: true },
      ]);
    } finally {
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
                <p className="answer">{message.text}</p>

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

                {message.citations && message.citations.length > 0 && (
                  <CitationList citations={message.citations} projectId={activeProjectId} />
                )}
              </div>
            </div>
          ),
        )}

        {pending && (
          <div className="turn turn--assistant">
            <div className="bubble bubble--assistant">
              <p className="muted">Retrieving passages and generating locally...</p>
            </div>
          </div>
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
        <button type="submit" className="btn btn--primary" disabled={pending || !input.trim() || !activeProjectId}>
          {pending ? "Thinking..." : "Ask"}
        </button>
      </form>
    </section>
  );
}
