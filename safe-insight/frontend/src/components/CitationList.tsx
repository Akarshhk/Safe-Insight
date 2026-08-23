/**
 * The citation trail under an answer.
 *
 * Each citation is a collapsed card showing the source, its location and the
 * retrieval score. Expanding one fetches the *untruncated* chunk from
 * `GET /chunks/{vector_id}` - the excerpt in the answer payload is display-sized,
 * and the point of this feature is that the user can always get to the exact
 * text the model read, not a summary of it.
 *
 * Citations the answer actually cited (`referenced`) are highlighted; the rest
 * were retrieved but unused, which is worth showing rather than hiding - it is
 * evidence about what the model had available.
 */

import { useState } from "react";
import { getChunk } from "../api";
import type { Citation, ChunkDetail } from "../types";

interface Props {
  citations: Citation[];
}

export default function CitationList({ citations }: Props) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [details, setDetails] = useState<Record<number, ChunkDetail>>({});
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (citations.length === 0) return null;

  async function toggle(citation: Citation) {
    if (openIndex === citation.index) {
      setOpenIndex(null);
      return;
    }
    setOpenIndex(citation.index);
    setError(null);

    // Fetch the full chunk once, then serve it from the local cache.
    if (details[citation.vector_id]) return;
    setLoadingId(citation.vector_id);
    try {
      const detail = await getChunk(citation.vector_id);
      setDetails((current) => ({ ...current, [citation.vector_id]: detail }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="citations">
      <h4 className="citations__title">
        Sources ({citations.length})
        <span className="muted small"> · click to read the exact source text</span>
      </h4>

      <ol className="citations__list">
        {citations.map((citation) => {
          const isOpen = openIndex === citation.index;
          const detail = details[citation.vector_id];
          return (
            <li
              key={citation.vector_id}
              className={`citation ${citation.referenced ? "citation--used" : ""}`}
            >
              <button
                type="button"
                className="citation__head"
                onClick={() => void toggle(citation)}
                aria-expanded={isOpen}
              >
                <span className="citation__marker">[{citation.index}]</span>
                <span className="citation__location">{citation.location}</span>
                <span className="citation__score" title="Cosine similarity to the question">
                  {citation.score.toFixed(3)}
                </span>
                {citation.referenced && (
                  <span className="citation__tag" title="Cited in the answer text">
                    cited
                  </span>
                )}
                <span className="citation__chevron">{isOpen ? "−" : "+"}</span>
              </button>

              {!isOpen && <p className="citation__excerpt muted">{citation.excerpt}</p>}

              {isOpen && (
                <div className="citation__body">
                  {loadingId === citation.vector_id && <p className="muted">Loading source...</p>}
                  {error && <p className="status status--error">{error}</p>}
                  {detail && (
                    <>
                      <p className="muted small">
                        {detail.location} · chunk #{detail.chunk_index + 1} · vector id{" "}
                        {detail.vector_id}
                      </p>
                      <blockquote className="citation__source">{detail.text}</blockquote>
                    </>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
