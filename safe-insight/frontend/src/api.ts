/**
 * Thin client for the local FastAPI backend.
 *
 * Every request goes to 127.0.0.1 - loopback only. That is the single network
 * hop the whole product allows, and it never leaves the machine. There is no
 * base-URL configuration for a remote host by design: if you find yourself
 * adding one, you are breaking the core guarantee.
 */

import type {
  ChunkDetail,
  HealthResponse,
  IngestResponse,
  OfflineStatus,
  QueryResponse,
} from "./types";

/** Keep this port in sync with `SAFE_INSIGHT_PORT` / `config.PORT` in Python. */
export const BACKEND_PORT = 8765;
export const BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;

/** Error carrying the backend's HTTP status, so callers can branch on it. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Fetch wrapper that unwraps FastAPI's `{ "detail": ... }` error envelope.
 *
 * A failed `fetch` (as opposed to a non-2xx response) almost always means the
 * Python process is not up yet, so we translate it into a message that says so
 * rather than the browser's opaque "Failed to fetch".
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "Cannot reach the Safe Insight backend on 127.0.0.1:" +
        BACKEND_PORT +
        ". It may still be starting up, or it failed to launch - check the terminal.",
      0,
    );
  }

  if (!response.ok) throw await errorFromResponse(response);

  return (await response.json()) as T;
}

/** Turn a non-2xx response into an ApiError carrying FastAPI's `detail`. */
async function errorFromResponse(response: Response): Promise<ApiError> {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") detail = body.detail;
  } catch {
    // Non-JSON error body; keep the status line we already have.
  }
  return new ApiError(detail, response.status);
}

/** Full backend status - index counts, model state, audit stats, offline check. */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/**
 * Offline verification for the status badge.
 *
 * @param rerun When true the backend runs a *live* outbound probe instead of
 *   returning its cached startup result. This is the button to press during a
 *   demo right after switching the machine to airplane mode.
 */
export function getOfflineStatus(rerun = false): Promise<OfflineStatus> {
  return request<OfflineStatus>(`/health/offline-check${rerun ? "?rerun=true" : ""}`);
}

export function listProjects(): Promise<any> {
  return request("/projects");
}

export function createProject(name: string): Promise<any> {
  return request("/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function getProject(projectId: string): Promise<any> {
  return request(`/projects/${encodeURIComponent(projectId)}`);
}

export function deleteProject(projectId: string): Promise<any> {
  return request(`/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" });
}

export function uploadDocument(projectId: string, file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<IngestResponse>(`/projects/${encodeURIComponent(projectId)}/documents`, { method: "POST", body: form });
}

export function deleteDocument(projectId: string, docId: string): Promise<{ doc_id: string; removed_vectors: number }> {
  return request(`/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
}

/** Ask a question. Retrieval + generation + audit logging happen server-side. */
export function askQuestion(projectId: string, question: string, topK: number, strictMode: boolean): Promise<QueryResponse> {
  return request<QueryResponse>(`/projects/${encodeURIComponent(projectId)}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, strict_mode: strictMode }),
  });
}

/** One line of the streaming endpoint's newline-delimited JSON. */
type StreamEvent =
  | { type: "token"; text: string }
  | { type: "ping" }
  | ({ type: "done" } & QueryResponse)
  | { type: "error"; detail: string };

/**
 * The backend pings every 3 s while it has nothing else to send (e.g. while the
 * model reads a long prompt), so this much silence means it has gone away.
 */
const STREAM_SILENCE_MS = 20_000;

/**
 * Ask a question and receive the answer while it is being written.
 *
 * Built never to hang: silence beyond STREAM_SILENCE_MS, a dropped connection,
 * or a stream that ends without its final event all reject with an ApiError.
 * Aborting `signal` (the Stop button) rejects too - and closing the connection
 * makes the backend stop generating, so the next question is not kept waiting.
 *
 * Resolves with the same body as `askQuestion`; its `answer` is authoritative
 * (it can carry an end note such as "[Answer cut off at the length limit.]").
 */
export async function askQuestionStream(
  projectId: string,
  question: string,
  topK: number,
  strictMode: boolean,
  onToken: (text: string) => void,
  signal: AbortSignal,
): Promise<QueryResponse> {
  const controller = new AbortController();
  const stopFromCaller = () => controller.abort();
  signal.addEventListener("abort", stopFromCaller);
  let silence: ReturnType<typeof setTimeout> | undefined;
  let silent = false;
  const resetSilence = () => {
    clearTimeout(silence);
    silence = setTimeout(() => {
      silent = true;
      controller.abort();
    }, STREAM_SILENCE_MS);
  };

  try {
    resetSilence();
    const response = await fetch(
      `${BASE_URL}/projects/${encodeURIComponent(projectId)}/query/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK, strict_mode: strictMode }),
        signal: controller.signal,
      },
    );
    if (!response.ok || !response.body) throw await errorFromResponse(response);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      resetSilence();
      buffered += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        const line = buffered.slice(0, newline).trim();
        buffered = buffered.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line) as StreamEvent;
        if (event.type === "token") onToken(event.text);
        else if (event.type === "done") return event;
        else if (event.type === "error") throw new ApiError(event.detail, 500);
      }
    }
    throw new ApiError("The connection to the backend closed before the answer finished.", 0);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (silent) {
      throw new ApiError("The backend stopped responding, so the answer was abandoned.", 0);
    }
    if (signal.aborted) throw error; // the caller's own Stop
    throw new ApiError(
      "Lost the connection to the Safe Insight backend. It may have stopped - check the terminal.",
      0,
    );
  } finally {
    clearTimeout(silence);
    signal.removeEventListener("abort", stopFromCaller);
  }
}

/** Untruncated source text behind a citation - fetched when the user expands it. */
export function getChunk(projectId: string, vectorId: number): Promise<ChunkDetail> {
  return request<ChunkDetail>(`/projects/${encodeURIComponent(projectId)}/chunks/${vectorId}`);
}

/** Verify the audit log's hash chain (tamper check). */
export function verifyAudit(): Promise<{
  valid: boolean;
  checked: number;
  total: number;
  detail: string;
  broken_at_id: number | null;
}> {
  return request("/audit/verify");
}

/**
 * Poll `/health` until the backend answers.
 *
 * The Tauri shell spawns Python and opens the window immediately, so the UI is
 * live a few seconds before the backend is. This turns that race into a
 * "Starting local engine..." state instead of a wall of failed requests.
 */
export async function waitForBackend(
  timeoutMs = 90_000,
  intervalMs = 750,
): Promise<HealthResponse> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown;

  while (Date.now() < deadline) {
    try {
      return await getHealth();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new ApiError("Backend did not start in time.", 0);
}

export function getSetupStatus(): Promise<{ model_present: boolean; active_model: string | null }> {
  return request("/setup/status");
}

export function getModelCatalog(): Promise<any[]> {
  return request("/setup/catalog");
}

export function setActiveModel(filename: string): Promise<any> {
  return request("/setup/active-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
}

export function switchModel(filename: string): Promise<any> {
  return request("/setup/switch-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
}
