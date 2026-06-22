"""CLI `ccgram status` — show running state without bot token.

Reads state files and the active multiplexer backend to display:
  - ccgram version
  - Backend session info (tmux session / herdr panes, window count)
  - Per-window status: bound/unbound, alive/dead

Multiplexer-aware: ``CCGRAM_MULTIPLEXER`` (default ``tmux``) selects the
backend, mirroring ``doctor_cmd``. The session_map key prefix and the live
window listing both follow that choice so herdr keys (``herdr:wN:pM``) are
counted and herdr panes are listed.

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
from .window_resolver import session_map_prefix_for

_TMUX_FORMAT_PARTS = 2

# Multiplexer backend selection (mirrors config.multiplexer_name; status reads
# the env directly to keep its "no Config import" startup contract, like doctor).
_MULTIPLEXER_ENV = "CCGRAM_MULTIPLEXER"
_DEFAULT_MULTIPLEXER = "tmux"
_HERDR_BACKEND = "herdr"


def _active_multiplexer_name() -> str:
    """Return the configured multiplexer backend (``CCGRAM_MULTIPLEXER``)."""
    return os.environ.get(_MULTIPLEXER_ENV, _DEFAULT_MULTIPLEXER)


def _list_herdr_windows() -> list[dict[str, str]]:
    """List herdr agent panes via the neutral seam. Returns list of {id, name}.

    Best-effort like ``_list_tmux_windows``: degrades to an empty list when the
    herdr socket is unreachable or the backend errors, so ``ccgram status``
    still prints state-file data.
    """
    # Lazy: the registry lazy-imports the backend; defer to keep status startup
    # light and touch only the neutral seam (never a concrete backend, F1).
    from .multiplexer import get_multiplexer

    try:
        windows = asyncio.run(get_multiplexer(_HERDR_BACKEND).list_windows())
    except Exception:  # noqa: BLE001 — status is best-effort; degrade to empty
        return []
    return [{"id": w.window_id, "name": w.window_name} for w in windows]


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict on any error."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return {}


def _list_tmux_windows(session_name: str) -> list[dict[str, str]]:
    """List tmux windows via subprocess. Returns list of {id, name}."""
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
            return []
        windows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == _TMUX_FORMAT_PARTS:
                windows.append({"id": parts[0], "name": parts[1]})
        return windows
    except (OSError, subprocess.TimeoutExpired):  # fmt: skip
        return []


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
        backend_line = f"Herdr: {len(live_windows)} pane(s)"
    else:
        live_windows = _list_tmux_windows(session_name)
        backend_line = f"Tmux session: {session_name} ({len(live_windows)} windows)"

    # Build binding index: window_id -> (thread_id, user_id)
    thread_bindings = state.get("thread_bindings", {})
    display_names = state.get("window_display_names", {})
    bound_windows: dict[str, tuple[int, int]] = {}
    for user_id_str, bindings in thread_bindings.items():
        for thread_id_str, window_id in bindings.items():
            bound_windows[window_id] = (int(thread_id_str), int(user_id_str))

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
    for w in live_windows:
        wid = w["id"]
        name = display_names.get(wid, w["name"])
        shown_ids.add(wid)

        if wid in bound_windows:
            thread_id, user_id = bound_windows[wid]
            print(
                f"  {wid:<5} {name:<16} -> topic {thread_id} (user {user_id})   alive"
            )
        else:
            print(f"  {wid:<5} {name:<16}                              (unbound)")

    # Show dead bindings (bound but window gone)
    for wid, (thread_id, user_id) in bound_windows.items():
        if wid not in shown_ids:
            name = display_names.get(wid, wid)
            print(f"  {wid:<5} {name:<16} -> topic {thread_id} (user {user_id})   dead")

    sys.exit(0)
