/**
 * "Offline Mode: Verified" status badge.
 *
 * Polls `GET /health/offline-check` and renders one of three states:
 *
 * - **Verified** - the socket guard is installed, strict, and its live outbound
 *   probe was blocked; nothing has tried to escape since the process started.
 * - **Not verified** - the guard reports a problem, or something *did* attempt
 *   an outbound connection. Loud on purpose: a false green badge on a privacy
 *   product is worse than no badge.
 * - **Checking** - initial load or a manual re-check in flight.
 *
 * Clicking the badge forces a live re-probe (`?rerun=true`). That is the demo
 * gesture: turn on airplane mode, click the badge, watch it stay green.
 */

import { useCallback, useEffect, useState } from "react";
import { getOfflineStatus } from "../api";
import type { OfflineStatus } from "../types";

const POLL_INTERVAL_MS = 15_000;

export default function OfflineBadge() {
  const [status, setStatus] = useState<OfflineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const refresh = useCallback(async (rerun: boolean) => {
    setChecking(true);
    try {
      setStatus(await getOfflineStatus(rerun));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void refresh(false);
    const timer = setInterval(() => void refresh(false), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const verified = status?.offline_verified === true;
  const downloading = status?.download_in_progress === true;
  const state = error ? "error" : checking && !status ? "checking" : downloading ? "setup" : verified ? "ok" : "warn";

  const label =
    state === "checking"
      ? "Checking offline status..."
      : state === "error"
        ? "Offline status unavailable"
        : downloading
          ? "Setup: Downloading model"
          : verified
            ? "Offline Mode: Verified"
            : "Offline Mode: NOT verified";

  return (
    <div className="offline-badge-wrap">
      <button
        type="button"
        className={`offline-badge offline-badge--${state}`}
        onClick={() => {
          setExpanded((open) => !open);
          void refresh(true);
        }}
        title="Click to re-run the live outbound-connection probe"
      >
        <span className="offline-badge__dot" aria-hidden="true" />
        {label}
        {verified ? " ✅" : ""}
      </button>

      {expanded && (
        <div className="offline-badge__detail">
          {error ? (
            <p className="muted">{error}</p>
          ) : status ? (
            <>
              <dl>
                <dt>Socket guard installed</dt>
                <dd>{status.guard_installed ? "yes" : "no"}</dd>
                <dt>Download in progress</dt>
                <dd>{status.download_in_progress ? "yes (sanctioned network exception)" : "no"}</dd>
                <dt>Strict mode</dt>
                <dd>{status.strict_mode ? "yes (connections raise)" : "no (audit only)"}</dd>
                <dt>Outbound probe</dt>
                <dd>{status.outbound_probe_blocked ? "blocked" : "NOT blocked"}</dd>
                <dt>Escape attempts (all blocked)</dt>
                <dd>{status.violation_count}</dd>
                <dt>HTTP client modules loaded</dt>
                <dd>
                  {status.http_client_modules_loaded.length === 0
                    ? "none"
                    : status.http_client_modules_loaded.join(", ")}
                </dd>
              </dl>
              {status.violation_count > 0 && (
                <>
                  {/*
                    A blocked attempt is the guard working, not failing - but it
                    turns the badge red on purpose, because the user deserves to
                    know something in the process tried to reach the network.
                    Naming the host makes it diagnosable: attempts to
                    huggingface.co almost always mean the embedding model was
                    never cached, so run download_model.py.
                  */}
                  <p className="status status--error small">
                    Blocked:{" "}
                    {Array.from(new Set(status.recent_violations.map((v) => v.host))).join(", ")}
                  </p>
                </>
              )}
              <p className="muted small">{status.self_check_detail}</p>
            </>
          ) : (
            <p className="muted">Running probe...</p>
          )}
        </div>
      )}
    </div>
  );
}
