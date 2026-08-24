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
  DocumentListResponse,
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

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error body; keep the status line we already have.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
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

export function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/documents"); // Note: there is no global /documents endpoint anymore, we need to remove this or scope it
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
