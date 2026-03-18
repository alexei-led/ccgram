"""Shared utility functions used across multiple CCGram modules.

Provides:
  - ccgram_dir(): resolve config directory from CCBOT_DIR env var.
  - tmux_session_name(): resolve tmux session name from env.
  - atomic_write_json(): crash-safe JSON file writes via temp+rename.
  - read_cwd_from_jsonl(): extract the cwd field from the first JSONL entry.
  - read_session_metadata_from_jsonl(): single-pass extraction of (cwd, summary).
  - task_done_callback(): log unhandled exceptions from background asyncio tasks.
  - log_throttled(): suppress repeated identical debug messages per key.
  - detect_tmux_context(): auto-detect tmux session name and own window ID.
  - check_duplicate_ccgram(): check if another ccgram is running in the session.
"""

import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# --- Log throttling -----------------------------------------------------------

_throttle_state: dict[str, tuple[float, str]] = {}


def log_throttled(
    log: Any,
    key: str,
    msg: str,
    *args: object,
    cooldown: float = 300.0,
    _clock: Callable[[], float] = time.monotonic,
) -> None:
    """Log at debug level, suppressing repeated identical messages per key.

    First occurrence always logs. Subsequent calls with the same *key* and
    identical formatted message are suppressed until *cooldown* seconds elapse.
    A changed message resets the timer and logs immediately.
    """
    formatted = msg % args if args else msg
    now = _clock()
    prev = _throttle_state.get(key)
    if prev and prev[1] == formatted and (now - prev[0]) < cooldown:
        return
    _throttle_state[key] = (now, formatted)
    log.debug(msg, *args)


def log_throttle_reset(prefix: str) -> None:
    """Clear throttle state for keys starting with *prefix*."""
    to_remove = [k for k in _throttle_state if k.startswith(prefix)]
    for k in to_remove:
        del _throttle_state[k]


def log_throttle_sweep(
    max_age: float = 600.0,
    _clock: Callable[[], float] = time.monotonic,
) -> int:
    """Remove throttle entries older than *max_age* seconds.

    Returns the number of entries removed.  Intended to be called
    periodically (e.g. every 60 s from the poll loop) to prevent
    unbounded growth of ``_throttle_state``.
    """
    now = _clock()
    stale = [k for k, (ts, _) in _throttle_state.items() if now - ts >= max_age]
    for k in stale:
        del _throttle_state[k]
    return len(stale)


CCGRAM_DIR_ENV = "CCGRAM_DIR"
_LEGACY_DIR_ENV = "CCBOT_DIR"

# Maximum number of JSONL lines to scan when extracting session metadata.
_SCAN_LINES = 20

_SUMMARY_MAX_CHARS = 80


def ccgram_dir() -> Path:
    """Resolve config directory from CCGRAM_DIR env var or default ~/.ccgram.

    Falls back to legacy CCBOT_DIR env var with a deprecation warning.
    If ~/.ccgram doesn't exist but ~/.ccbot does, logs a migration hint.
    """
    raw = os.environ.get(CCGRAM_DIR_ENV, "")
    if not raw:
        raw = os.environ.get(_LEGACY_DIR_ENV, "")
        if raw:
            logger.warning(
                "CCBOT_DIR is deprecated, use CCGRAM_DIR instead",
                old="CCBOT_DIR",
                new="CCGRAM_DIR",
            )
    if raw:
        return Path(raw)

    default = Path.home() / ".ccgram"
    legacy = Path.home() / ".ccbot"
    # Use legacy ~/.ccbot if it has a .env and ~/.ccgram does not
    if not (default / ".env").is_file() and (legacy / ".env").is_file():
        logger.warning(
            "Using legacy ~/.ccbot config dir. Migrate with: mv ~/.ccbot ~/.ccgram"
        )
        return legacy
    return default


def tmux_session_name() -> str:
    """Get tmux session name from TMUX_SESSION_NAME env var or default 'ccgram'."""
    return os.environ.get("TMUX_SESSION_NAME", "ccgram")


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data to a file atomically.

    Writes to a temporary file in the same directory, then renames it
    to the target path. This prevents data corruption if the process
    is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=indent)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=f".{path.name}."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def read_cwd_from_jsonl(file_path: str | Path) -> str:
    """Read the cwd field from the first JSONL entry that has one.

    Scans up to _SCAN_LINES lines. Shared by session.py and session_monitor.py.
    """
    cwd, _ = read_session_metadata_from_jsonl(file_path)
    return cwd


def _extract_user_text(msg: dict[str, object]) -> str:
    """Extract display text from a user message's content field."""
    content = msg.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text:
                    return text[:_SUMMARY_MAX_CHARS]
    elif isinstance(content, str) and content:
        return content[:_SUMMARY_MAX_CHARS]
    return ""


def _extract_metadata_from_entry(data: dict, cwd: str, summary: str) -> tuple[str, str]:
    """Extract cwd and summary fields from a single parsed JSONL entry."""
    if not cwd:
        found_cwd = data.get("cwd")
        if found_cwd and isinstance(found_cwd, str):
            cwd = found_cwd
    if not summary and data.get("type") == "user":
        msg = data.get("message", {})
        if isinstance(msg, dict):
            summary = _extract_user_text(msg)
    return cwd, summary


def read_session_metadata_from_jsonl(file_path: str | Path) -> tuple[str, str]:
    """Extract cwd and summary from a JSONL transcript in a single file read.

    Scans up to _SCAN_LINES lines. Returns (cwd, summary) where either
    may be empty if not found.
    """
    cwd = ""
    summary = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= _SCAN_LINES:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                cwd, summary = _extract_metadata_from_entry(data, cwd, summary)
                if cwd and summary:
                    break
    except OSError:
        pass
    return cwd, summary


def detect_tmux_context() -> tuple[str | None, str | None]:
    """Detect tmux session name and own window ID in a single tmux call.

    Returns (session_name, own_window_id). Either may be None.
    Requires $TMUX to be set (running inside tmux).
    """
    if not os.environ.get("TMUX"):
        return None, None
    pane_id = os.environ.get("TMUX_PANE")
    if not pane_id:
        # TMUX set but no TMUX_PANE — can only get session name
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "#{session_name}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None, None
            name = result.stdout.strip()
            return (name or None), None
        except subprocess.TimeoutExpired, FileNotFoundError:
            return None, None
    # Single call to get both session name and window ID
    try:
        result = subprocess.run(
            [
                "tmux",
                "display-message",
                "-t",
                pane_id,
                "-p",
                "#{session_name}\t#{window_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None, None
        parts = result.stdout.strip().split("\t", 1)
        session_name = parts[0] if parts[0] else None
        window_id = parts[1] if len(parts) > 1 and parts[1] else None
        return session_name, window_id
    except subprocess.TimeoutExpired, FileNotFoundError:
        return None, None


def check_duplicate_ccgram(session_name: str) -> str | None:
    """Check if another ccgram is running in the session.

    Returns error message if duplicate found, None if clear.
    """
    own_pane = os.environ.get("TMUX_PANE", "")
    if not own_pane:
        # Cannot reliably exclude self — skip duplicate check
        return None
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-s",
                "-t",
                session_name,
                "-F",
                "#{pane_id}\t#{window_id}\t#{pane_current_command}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired, FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:  # noqa: PLR2004
            continue
        pane_id, window_id, cmd = parts
        if pane_id == own_pane:
            continue
        if cmd.strip() == "ccgram":
            return (
                f"Another ccgram instance is already running in "
                f"tmux session '{session_name}' (window {window_id})"
            )
    return None


def task_done_callback(task: asyncio.Task[None]) -> None:
    """Log unhandled exceptions from background asyncio tasks.

    Attach to any fire-and-forget task via ``task.add_done_callback(task_done_callback)``.
    Suppresses CancelledError (normal shutdown).
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background task %s failed", task.get_name(), exc_info=exc)
