/**
 * Wire types shared with the Python backend.
 *
 * These mirror `backend/app/schemas.py` field for field. If you change a model
 * there, change its twin here - nothing enforces the match automatically, so it
 * is worth keeping the two files open side by side.
 */

/** One indexed document, as listed in the document panel. */
export interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  sha256: string;
  page_count: number | null;
  chunk_count: number;
  ingested_at: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total_chunks: number;
}

export interface IngestResponse {
  doc_id: string;
  filename: string;
  chunk_count: number;
  page_count: number | null;
  /** Set when an identical file was already indexed; nothing new was added. */
  duplicate_of: string | null;
  elapsed_ms: number;
}

/**
 * One entry in the citation trail.
 *
 * `index` is 1-based and matches the `[n]` markers the model writes into the
 * answer, so `[2]` always means the citation with `index === 2`.
 */
export interface Citation {
  index: number;
  vector_id: number;
  doc_id: string;
  filename: string;
  page_number: number | null;
  chunk_index: number;
  excerpt: string;
  truncated: boolean;
  score: number;
  location: string;
  /** True when the answer text actually cited this source. */
  referenced: boolean;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  /** False when retrieval found nothing above the similarity floor. */
  answered_from_context: boolean;
  /** True when answer is grounded, false when it's general knowledge. */
  grounded: boolean;
  /** True when no GGUF model is loaded and this is the retrieval-only stub. */
  stub_mode: boolean;
  model: string;
  embedding_model: string | null;
  top_k: number;
  retrieved_count: number;
  latency_ms: number;
  audit_id: number;
}

/** Full, untruncated source text behind a citation. */
export interface ChunkDetail {
  vector_id: number;
  doc_id: string;
  filename: string;
  page_number: number | null;
  chunk_index: number;
  text: string;
  location: string;
}

/** One blocked outbound connection attempt, recorded by the socket guard. */
export interface NetworkViolation {
  /** Seconds since the epoch. */
  timestamp: number;
  host: string;
  port: number | null;
  /** Which patched call was used, e.g. "socket.connect", "socket.getaddrinfo". */
  action: string;
}

/** Payload behind the "Offline Mode: Verified" badge. */
export interface OfflineStatus {
  offline_verified: boolean;
  guard_installed: boolean;
  strict_mode: boolean;
  outbound_probe_blocked: boolean | null;
  self_check_detail: string | null;
  violation_count: number;
  recent_violations: NetworkViolation[];
  http_client_modules_loaded: string[];
  download_in_progress?: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  offline: OfflineStatus;
  index: {
    documents: number;
    chunks: number;
    vectors: number;
    dimension: number | null;
    embedding_model: string | null;
    index_path: string;
    loaded: boolean;
  };
  llm: {
    loaded: boolean;
    model_file: string;
    model_present: boolean;
    load_error: string | null;
    [key: string]: unknown;
  };
  audit: {
    query_count: number;
    document_event_count: number;
    last_query_at: string | null;
    database_path: string;
    [key: string]: unknown;
  };
  embedding_model_loaded: boolean;
}

/** A turn in the chat transcript. Errors are rendered as assistant turns. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  meta?: {
    latencyMs: number;
    model: string;
    retrievedCount: number;
    stubMode: boolean;
    auditId: number;
    grounded: boolean;
  };
  isError?: boolean;
}
