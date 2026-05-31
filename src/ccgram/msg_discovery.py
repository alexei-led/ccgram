"""Peer discovery — view over SessionManager state + self-declared overlay.

NOT a separate registry file. Discovery reads from SessionManager window_states
(auto fields: window_id, name, provider, cwd) and a small ``declared.json``
overlay (task, team). Branch is detected via ``git rev-parse`` in the window's cwd.

Key functions: list_peers(), register_declared(), clear_declared().
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import structlog

from .config import config
from .topic_state_registry import topic_state
from .utils import atomic_write_json, ccgram_dir

logger = structlog.get_logger()


@dataclass
class WindowInfo:
    """Lightweight window state for peer discovery (no config dependency)."""

    cwd: str = ""
    window_name: str = ""
    provider_name: str = ""


@dataclass
class PeerInfo:
    """Merged view of a peer agent window."""

    window_id: str
    name: str
    provider: str
    cwd: str
    branch: str
    task: str
    team: str


def export_window_info() -> dict[str, WindowInfo]:
    """CLI-safe snapshot of window states. Reads state.json from disk.

    Returns {window_id: WindowInfo} without requiring a bot token or
    SessionManager initialization. Used by ``ccgram msg`` CLI commands.
    Uses ccgram_dir() so callers can patch the directory for testing.
    """
    state_file = ccgram_dir() / "state.json"
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text())
    except json.JSONDecodeError, OSError:
        return {}
    result: dict[str, WindowInfo] = {}
    for window_id, ws_data in data.get("window_states", {}).items():
        if isinstance(ws_data, dict):
            result[window_id] = WindowInfo(
                cwd=ws_data.get("cwd", ""),
                window_name=ws_data.get("window_name", ""),
                provider_name=ws_data.get("provider_name", ""),
            )
    return result


def _default_declared_path() -> Path:
    return config.mailbox_dir / "declared.json"


def _load_declared_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError, OSError:
        return {}


def _save_declared_file(path: Path, data: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)


def detect_branch(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return ""


def _qualify_window_id(window_id: str, tmux_session: str) -> str:
    if ":" in window_id:
        return window_id
    return f"{tmux_session}:{window_id}"


def register_declared(
    window_id: str,
    *,
    task: str | None = None,
    team: str | None = None,
    path: Path | None = None,
) -> None:
    """Update self-declared overlay for a window (task, team)."""
    resolved_path = path if path is not None else _default_declared_path()
    data = _load_declared_file(resolved_path)
    entry = data.get(window_id, {})
    if task is not None:
        entry["task"] = task
    if team is not None:
        entry["team"] = team
    data[window_id] = entry
    _save_declared_file(resolved_path, data)


@topic_state.register("qualified")
def clear_declared(window_id: str, *, path: Path | None = None) -> None:
    """Remove self-declared overlay entry on window death."""
    resolved_path = path if path is not None else _default_declared_path()
    data = _load_declared_file(resolved_path)
    if window_id not in data:
        return
    del data[window_id]
    _save_declared_file(resolved_path, data)


def list_peers(
    *,
    window_states: dict[str, WindowInfo],
    tmux_session: str,
    declared_path: Path | None = None,
    filter_provider: str | None = None,
    filter_team: str | None = None,
    filter_cwd: str | None = None,
) -> list[PeerInfo]:
    """Build merged peer list from window states + declared overlay.

    Args:
        window_states: WindowState dict from SessionManager (window_id -> state).
        tmux_session: Current tmux session name for qualifying bare window IDs.
        declared_path: Path to declared.json (default: config mailbox_dir).
        filter_provider: Only include peers with this provider.
        filter_team: Only include peers with this team (from declared overlay).
        filter_cwd: Only include peers whose cwd matches this glob pattern.

    Returns:
        List of PeerInfo sorted by qualified window ID.
    """
    resolved_path = (
        declared_path if declared_path is not None else _default_declared_path()
    )
    declared = _load_declared_file(resolved_path)

    peers: list[PeerInfo] = []
    for window_id, ws in window_states.items():
        qualified_id = _qualify_window_id(window_id, tmux_session)
        decl = declared.get(qualified_id, {})

        if filter_provider and ws.provider_name != filter_provider:
            continue
        if filter_team and decl.get("team", "") != filter_team:
            continue
        if filter_cwd and not fnmatch(ws.cwd, filter_cwd):
            continue

        branch = detect_branch(ws.cwd) if ws.cwd else ""

        peers.append(
            PeerInfo(
                window_id=qualified_id,
                name=ws.window_name,
                provider=ws.provider_name,
                cwd=ws.cwd,
                branch=branch,
                task=decl.get("task", ""),
                team=decl.get("team", ""),
            )
        )

    peers.sort(key=lambda p: p.window_id)
    return peers
