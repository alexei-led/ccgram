"""CLI `ccgram status` — show running state without bot token.

Reads state files and the active multiplexer backend to display:
  - ccgram version
  - Backend session info (tmux session / herdr guarded sessions, window count)
  - Per-window status: bound/unbound, alive/dead

Multiplexer-aware: ``CCGRAM_MULTIPLEXER`` (default ``tmux``) selects the
backend, mirroring ``doctor_cmd``. The session_map key prefix and the live
window listing both follow that choice; Herdr bindings use opaque
``herdr-session-v1-*`` targets.

No Config import needed — loads ``~/.ccgram/.env`` (and a local ``.env``) via
``utils.load_ccgram_env`` so ``CCGRAM_MULTIPLEXER`` set only in the config-dir
``.env`` is honored, then reads it directly and uses utils.ccgram_dir().
``providers.resolve_capabilities``, the package ``__version__``, and the herdr
backend (via the neutral seam) are imported lazily inside the subcommand body to
keep ``ccgram --help`` free of provider-registry initialization.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from .utils import ccgram_dir, load_ccgram_env, tmux_session_name
from .multiplexer.base import canonical_window_id
from .window_resolver import session_map_prefix_for

_TMUX_FORMAT_PARTS = 2

# Multiplexer backend selection (mirrors config.multiplexer_name; status reads
# the env directly to keep its "no Config import" startup contract, like doctor).
_MULTIPLEXER_ENV = "CCGRAM_MULTIPLEXER"
_DEFAULT_MULTIPLEXER = "tmux"
_HERDR_BACKEND = "herdr"
_TMUX_BACKEND = "tmux"


def _active_multiplexer_name() -> str:
    """Return the configured multiplexer backend (``CCGRAM_MULTIPLEXER``)."""
    return os.environ.get(_MULTIPLEXER_ENV, _DEFAULT_MULTIPLEXER)


def _list_herdr_windows() -> list[dict[str, str]] | None:
    """List herdr agent panes via the neutral seam, as {id, name}.

    ``None`` when the herdr socket is unreachable or the backend errors, like
    ``_list_tmux_windows``: not an empty listing, which the caller renders as
    every binding dead. ``ccgram status`` still prints its state-file data.
    """
    return _live_windows(_HERDR_BACKEND)


def _list_backend_windows(mux_name: str) -> list[dict[str, str]] | None:
    """List windows through the seam, for a backend with no bespoke listing.

    ``None`` when the backend could not be asked, like the herdr listing above:
    the caller renders an empty listing as every binding dead. Best-effort
    either way — status still prints its state-file data.
    """
    return _live_windows(mux_name)


def _live_windows(mux_name: str) -> list[dict[str, str]] | None:
    """List every live window on ``mux_name``, for the alive/dead column.

    Reads the reconciliation listing, not ``list_windows``: status reports
    liveness, and the selection listing applies the backend's visibility
    filters — the agterm workspace scope, a herdr internal workspace, tmux's
    hidden names. Reporting a bound session dead because it fell outside one
    of those would be wrong, and dead bindings are what /sync Fix offers to
    clean up.

    Returns ``None`` when the listing could not be confirmed — a backend
    error, or the backend's own ``None``. That is distinct from a confirmed
    empty listing: an outage means every window is unknown, and rendering
    unknown as ``dead`` is the same false claim this listing exists to
    prevent. The state-file data is still printed either way.
    """
    # Lazy: the registry lazy-imports the backend; defer to keep status startup
    # light and touch only the neutral seam (never a concrete backend, F1).
    from .multiplexer import get_multiplexer

    try:
        windows = asyncio.run(
            get_multiplexer(mux_name).list_windows_for_reconciliation()
        )
    except Exception:  # noqa: BLE001 — status is best-effort; unknown, not empty
        return None
    if windows is None:
        return None
    return [{"id": w.window_id, "name": w.window_name} for w in windows]


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict on any error."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return {}


def _list_tmux_windows(session_name: str) -> list[dict[str, str]] | None:
    """List tmux windows via subprocess, or ``None`` if tmux could not answer.

    A missing or unresponsive tmux is not an empty server: reporting it as one
    labels every binding dead. See ``_live_windows`` for the same distinction
    on the seam backends.
    """
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-windows",
                "-t",
                session_name,
                "-F",
                "#{window_id}\t#{window_name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        windows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == _TMUX_FORMAT_PARTS:
                windows.append({"id": parts[0], "name": parts[1]})
        return windows
    except (OSError, subprocess.TimeoutExpired):  # fmt: skip
        return None


def _capability_summary() -> tuple[str, str]:
    """Return (provider_name, comma-separated capability flags)."""
    # Lazy: keep `ccgram status` startup snappy
    from .providers import resolve_capabilities

    caps = resolve_capabilities()
    flags = [
        label
        for flag, label in (
            (caps.supports_hook, "hook"),
            (caps.supports_resume, "resume"),
            (caps.supports_continue, "continue"),
        )
        if flag
    ]
    return caps.name, ", ".join(flags) or "none"


def _backend_line(label: str, windows: list[dict[str, str]] | None, unit: str) -> str:
    """One-line backend summary; an unconfirmed listing says so, not zero."""
    if windows is None:
        return f"{label}: unreachable"
    return f"{label}: {len(windows)} {unit}"


def status_main() -> None:
    """Entry point for `ccgram status`."""
    # Honor CCGRAM_* (e.g. CCGRAM_MULTIPLEXER) set only in ~/.ccgram/.env,
    # like the bot does via Config — must run before _active_multiplexer_name().
    load_ccgram_env()
    # Lazy: keep `ccgram status` startup snappy
    from . import __version__

    provider_name, cap_flags = _capability_summary()
    config_dir = ccgram_dir()
    mux_name = _active_multiplexer_name()
    session_name = tmux_session_name()

    # Read state files
    state = _read_json(config_dir / "state.json")
    session_map = _read_json(config_dir / "session_map.json")

    # Get live windows from the active multiplexer backend
    if mux_name == _HERDR_BACKEND:
        live_windows = _list_herdr_windows()
        backend_line = _backend_line("Herdr", live_windows, "pane(s)")
    elif mux_name != _TMUX_BACKEND:
        # Any other backend answers through the seam; only tmux and herdr have
        # a bespoke listing, and routing a third backend into the tmux branch
        # reports its session count as zero.
        live_windows = _list_backend_windows(mux_name)
        backend_line = _backend_line(mux_name, live_windows, "session(s)")
    else:
        live_windows = _list_tmux_windows(session_name)
        backend_line = (
            f"Tmux session: {session_name} (unreachable)"
            if live_windows is None
            else f"Tmux session: {session_name} ({len(live_windows)} windows)"
        )

    # Build binding index: window_id -> (thread_id, user_id)
    thread_bindings = state.get("thread_bindings", {})
    display_names = state.get("window_display_names", {})
    bound_windows: dict[str, tuple[int, int, str]] = {}
    for user_id_str, bindings in thread_bindings.items():
        for thread_id_str, window_id in bindings.items():
            bound_windows[canonical_window_id(window_id)] = (
                int(thread_id_str),
                int(user_id_str),
                window_id,
            )
    display_names_by_id = {
        canonical_window_id(wid): name for wid, name in display_names.items()
    }

    # Count monitored sessions (backend-aware prefix)
    prefix = session_map_prefix_for(mux_name, session_name)
    monitored = sum(1 for k in session_map if k.startswith(prefix))

    # Output
    print(f"ccgram {__version__}")
    print(f"Provider: {provider_name} ({cap_flags})")
    print(backend_line)
    print(f"Monitored sessions: {monitored}")

    if not live_windows and not bound_windows:
        return

    print()

    # Show live windows first
    shown_ids: set[str] = set()
    for w in live_windows or []:
        wid = w["id"]
        key = canonical_window_id(wid)
        name = display_names_by_id.get(key, w["name"])
        shown_ids.add(key)

        if key in bound_windows:
            thread_id, user_id, _bound_wid = bound_windows[key]
            print(
                f"  {wid:<5} {name:<16} -> topic {thread_id} (user {user_id})   alive"
            )
        else:
            print(f"  {wid:<5} {name:<16}                              (unbound)")

    # Bound but not in the live listing. Only a confirmed listing makes that
    # "dead"; an unconfirmed one makes it unknown, and /sync Fix acts on dead.
    verdict = "unknown" if live_windows is None else "dead"
    for key, (thread_id, user_id, wid) in bound_windows.items():
        if key not in shown_ids:
            name = display_names_by_id.get(key, wid)
            print(
                f"  {wid:<5} {name:<16} -> topic {thread_id} "
                f"(user {user_id})   {verdict}"
            )

    sys.exit(0)
