"""
Network kill-switch.

Safe Insight promises that documents never leave the machine. A promise is not
an architecture, so this module *enforces* it at the lowest layer the Python
process controls: the socket API itself.

Three layers of defence, weakest to strongest:

1. **Environment**  - ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set in
   :mod:`app.config` so the ML libraries never phone home for model updates.
2. **Socket monkey-patch**  - ``socket.socket.connect`` /
   ``connect_ex`` / ``socket.create_connection`` and ``socket.getaddrinfo`` are
   wrapped. Any destination that is not loopback is refused (strict mode) or
   recorded (audit mode).
3. **Startup self-check**  - we deliberately *try* to reach a public IP at boot.
   If the attempt succeeds, the guard is broken and we say so loudly instead of
   quietly shipping a false "offline" badge.

The ``/health/offline-check`` endpoint in :mod:`app.main` serves the result of
:func:`get_status`, which is what drives the "Offline Mode: Verified" badge.

Why patch instead of just "not using requests"?
    Any transitive dependency could open a socket. Patching means we do not have
    to audit the dependency tree by hand - and the guard's violation log gives
    the viva a concrete artefact to point at.
"""

from __future__ import annotations

import ipaddress
import socket
import sys
import threading
import time
import contextvars
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Set

from app import config
from app import audit_log

# --------------------------------------------------------------------------- #
# Module state
# --------------------------------------------------------------------------- #

#: Hostnames that are always allowed. Only the local FastAPI <-> Tauri hop.
_ALLOWED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "",
    None,
}

#: Every blocked (or observed) outbound attempt, newest last. Bounded so a
#: misbehaving dependency in a retry loop cannot exhaust memory.
_MAX_VIOLATIONS = 200
_violations: List[Dict[str, Any]] = []
_violations_lock = threading.Lock()

_installed = False
_self_check: Dict[str, Any] = {"ran": False, "outbound_blocked": None, "detail": "not run"}

_sanctioned_domains: contextvars.ContextVar[Set[str]] = contextvars.ContextVar("sanctioned_domains", default=set())
_sanctioned_download_active = False
_global_sanctioned_domains: Set[str] = set()

# Saved originals, so the self-check can bypass its own patch.
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


class OutboundNetworkBlocked(RuntimeError):
    """Raised when code attempts a non-loopback connection in strict mode."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _record_violation(host: str, port: Any, action: str) -> None:
    """Append an attempted outbound connection to the in-memory violation log."""
    entry = {
        "timestamp": time.time(),
        "host": str(host),
        "port": port,
        "action": action,
    }
    with _violations_lock:
        _violations.append(entry)
        if len(_violations) > _MAX_VIOLATIONS:
            del _violations[: len(_violations) - _MAX_VIOLATIONS]


def _is_local_address(host: Any) -> bool:
    """
    True when ``host`` is loopback (or a non-IP special value we allow).

    Anything we cannot positively identify as loopback is treated as remote -
    fail closed, never fail open.
    """
    if host in _ALLOWED_HOSTS:
        return True
    if isinstance(host, bytes):
        host = host.decode("utf-8")
    if not isinstance(host, str):
        return False
    candidate = host.strip().strip("[]").lower()
    if candidate in _ALLOWED_HOSTS:
        return True
    
    sanctioned = _sanctioned_domains.get()
                
    if not sanctioned and _sanctioned_download_active:
        sanctioned = _global_sanctioned_domains
        
    if sanctioned:
        if candidate in sanctioned:
            return True
        # Check domain suffix (e.g., .hf.co)
        for domain in sanctioned:
            if domain.startswith(".") and candidate.endswith(domain):
                return True
    
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A hostname rather than a literal IP. Only *.localhost is trusted.
        return candidate.endswith(".localhost")


def _split_address(address: Any) -> Tuple[Any, Any]:
    """Normalise the many shapes a socket address can take into (host, port)."""
    if isinstance(address, (tuple, list)) and address:
        return address[0], (address[1] if len(address) > 1 else None)
    # AF_UNIX / AF_PIPE addresses are strings and are inherently local.
    return address, None


def _check_destination(address: Any, action: str) -> None:
    """Raise (strict) or record (audit) if ``address`` is not loopback."""
    # Unix domain sockets / named pipes are addressed by path and never leave
    # the machine, so they are allowed unconditionally.
    if isinstance(address, (str, bytes)):
        return

    host, port = _split_address(address)
    if _is_local_address(host):
        sanctioned = _sanctioned_domains.get()
        if sanctioned and host not in _ALLOWED_HOSTS:
            # We are allowing a sanctioned domain. Log it!
            # To avoid recursion or spamming, we could rate limit, but let's log it simply.
            audit_log.log_system_event(
                action="sanctioned_network_access",
                detail={"host": str(host), "port": port, "action": action, "reason": "setup model download"}
            )
        return

    _record_violation(host, port, action)
    if config.STRICT_NETWORK_GUARD:
        raise OutboundNetworkBlocked(
            f"Safe Insight network guard blocked an outbound connection to "
            f"{host}:{port} via {action}(). Safe Insight is offline-only; no "
            f"runtime code may open a remote socket."
        )


# --------------------------------------------------------------------------- #
# Patched socket entry points
# --------------------------------------------------------------------------- #
def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check_destination(address, "socket.connect")
    return _orig_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    try:
        _check_destination(address, "socket.connect_ex")
    except OutboundNetworkBlocked:
        # connect_ex reports failures as errno rather than exceptions; mimic
        # "network is unreachable" so callers degrade the way they expect to.
        return 101
    return _orig_connect_ex(self, address)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
    _check_destination(address, "socket.create_connection")
    return _orig_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Block DNS resolution of remote hostnames.

    Stopping resolution as well as connection means a leak fails fast and with a
    clear message, instead of hanging on a socket timeout.
    """
    if not _is_local_address(host):
        _record_violation(host, port, "socket.getaddrinfo")
        if config.STRICT_NETWORK_GUARD:
            raise OutboundNetworkBlocked(
                f"Safe Insight network guard blocked DNS resolution of {host!r}."
            )
            
    result = _orig_getaddrinfo(host, port, *args, **kwargs)
    
    sanctioned = _sanctioned_domains.get()
    if not sanctioned and _sanctioned_download_active:
        sanctioned = _global_sanctioned_domains
        
    if sanctioned and host not in _ALLOWED_HOSTS:
        for item in result:
            sockaddr = item[4]
            if isinstance(sockaddr, tuple) and len(sockaddr) >= 1:
                ip = sockaddr[0]
                sanctioned.add(str(ip))
                if _sanctioned_download_active:
                    _global_sanctioned_domains.add(str(ip))
                
    return result


def install_guard() -> None:
    """
    Activate the socket patches. Call once, as early as possible in startup.

    Idempotent: a second call is a no-op, so importing this from tests is safe.
    """
    global _installed
    if _installed:
        return
    socket.socket.connect = _guarded_connect          # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex    # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _guarded_getaddrinfo         # type: ignore[assignment]
    _installed = True


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def run_self_check(timeout: float = 1.5) -> Dict[str, Any]:
    """
    Prove the guard works by attempting a real outbound connection.

    Target is 8.8.8.8:53 as a literal IP so the test exercises the *connect*
    patch rather than merely tripping the DNS patch. Success of the attempt is a
    failure of the guard.

    Returns a dict describing the outcome; also cached for :func:`get_status`.
    """
    global _self_check
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(timeout)
    blocked = False
    detail = ""
    try:
        # Deliberately go through the *patched* API - that is what we are testing.
        probe.connect(("8.8.8.8", 53))
        detail = (
            "Outbound probe to 8.8.8.8:53 SUCCEEDED - the network guard is not "
            "in effect. Do not trust the offline badge until this is fixed."
        )
    except OutboundNetworkBlocked as exc:
        blocked = True
        detail = f"Guard active: {exc}"
    except OSError as exc:
        # No route / machine genuinely offline. Also a pass, but note the reason
        # so the viva demo can distinguish "guard works" from "wifi is simply off".
        blocked = True
        detail = f"Outbound probe failed at the OS level ({exc.__class__.__name__}: {exc})."
    finally:
        probe.close()

    _self_check = {
        "ran": True,
        "outbound_blocked": blocked,
        "detail": detail,
        "checked_at": time.time(),
    }
    return _self_check


def _loaded_http_clients() -> List[str]:
    """
    Report HTTP client libraries that are actually imported in this process.

    Presence is not proof of a leak (FastAPI's TestClient pulls in httpx, for
    example), but an empty list is a strong, checkable claim for the report.
    """
    watched = ("requests", "httpx", "aiohttp", "urllib3", "urllib.request")
    return [name for name in watched if name in sys.modules]


def get_violations() -> List[Dict[str, Any]]:
    """Snapshot of blocked outbound attempts since process start."""
    with _violations_lock:
        return list(_violations)


def get_status(rerun_self_check: bool = False) -> Dict[str, Any]:
    """
    Build the payload behind ``GET /health/offline-check``.

    ``offline_verified`` is the single boolean the UI badge reads: it is True
    only when the guard is installed, running strict, the outbound probe was
    blocked, and nothing has tried to escape since boot.
    """
    if rerun_self_check or not _self_check["ran"]:
        run_self_check()

    violations = get_violations()
    verified = bool(
        _installed
        and config.STRICT_NETWORK_GUARD
        and _self_check.get("outbound_blocked")
        and not violations
        and not _sanctioned_download_active
    )
    return {
        "offline_verified": verified,
        "guard_installed": _installed,
        "strict_mode": config.STRICT_NETWORK_GUARD,
        "outbound_probe_blocked": _self_check.get("outbound_blocked"),
        "self_check_detail": _self_check.get("detail"),
        "violation_count": len(violations),
        "recent_violations": violations[-10:],
        "http_client_modules_loaded": _loaded_http_clients(),
        "download_in_progress": _sanctioned_download_active,
    }


def describe() -> str:
    """One-line human summary, used in startup logging."""
    status = get_status()
    if status["download_in_progress"]:
        state = "DOWNLOADING MODEL"
    else:
        state = "VERIFIED" if status["offline_verified"] else "NOT VERIFIED"
    return f"Offline mode: {state} ({status['self_check_detail']})"

@contextmanager
def sanctioned_download(reason: str, allowed_domains: Set[str]):
    """
    Temporarily allow outbound access to specific domains for a sanctioned download.
    This uses contextvars to ensure the exception applies only to the current async context.
    It also sets a global fallback for thread boundaries (like anyio DNS resolution) where context is lost.
    """
    global _sanctioned_download_active, _global_sanctioned_domains
    _sanctioned_download_active = True
    _global_sanctioned_domains = set(allowed_domains)
    token = _sanctioned_domains.set(set(allowed_domains))
    audit_log.log_system_event(
        action="sanctioned_download_start",
        detail={"reason": reason, "allowed_domains": list(allowed_domains)}
    )
    try:
        yield
    finally:
        _sanctioned_domains.reset(token)
        _global_sanctioned_domains = set()
        _sanctioned_download_active = False
        audit_log.log_system_event(
            action="sanctioned_download_end",
            detail={"reason": reason}
        )


__all__ = [
    "OutboundNetworkBlocked",
    "install_guard",
    "run_self_check",
    "get_status",
    "get_violations",
    "describe",
    "sanctioned_download",
]
