"""Tmux session/window management via libtmux.

Wraps libtmux to provide async-friendly operations on a single tmux session:
  - list_windows / find_window_by_name: discover Claude Code windows.
  - capture_pane: read terminal content (plain or with ANSI colors).
  - send_keys: forward user input or control keys to a window.
  - create_window / kill_window: lifecycle management.
  - list_panes / capture_pane_by_id / send_keys_to_pane: pane-level ops.
  - Vim mode detection: auto-enter INSERT mode before sending text when
    Claude Code's /vim mode is active and the TUI is in NORMAL mode.

All blocking libtmux calls are wrapped in asyncio.to_thread().

Key class: TmuxManager (singleton instantiated as `tmux_manager`).
Module-level: _vim_state cache, _vim_locks for per-window send serialization.
"""

import asyncio
import structlog
import subprocess
from dataclasses import dataclass
from pathlib import Path

import libtmux
from libtmux.exc import LibTmuxException

from .config import config
from .window_resolver import EMDASH_SESSION_PREFIX as _EMDASH_PREFIX, is_foreign_window

logger = structlog.get_logger()

# ── Vim mode state cache ───────────────────────────────────────────────
# window_id → True (vim mode on) / False (vim mode off)
# Missing key = unknown, needs probe on first send.
_vim_state: dict[str, bool] = {}

# Per-window locks to serialize vim probe + send sequences,
# preventing interleaved keystrokes from concurrent send_keys() calls.
_vim_locks: dict[str, asyncio.Lock] = {}

# Delay between sending probe 'i' and recapturing pane (seconds).
_VIM_PROBE_DELAY = 0.12


def _has_insert_indicator(pane_text: str) -> bool:
    """Check if ``-- INSERT --`` appears in the last 3 lines of pane text."""
    return any("-- INSERT --" in line for line in pane_text.splitlines()[-3:])


def notify_vim_insert_seen(window_id: str) -> None:
    """Record that vim INSERT mode was observed (called from status polling)."""
    _vim_state[window_id] = True


def clear_vim_state(window_id: str) -> None:
    """Remove vim state cache entry and lock for a window (called on cleanup)."""
    _vim_state.pop(window_id, None)
    _vim_locks.pop(window_id, None)


def reset_vim_state() -> None:
    """Reset all vim state (for testing)."""
    _vim_state.clear()
    _vim_locks.clear()


_TmuxError = (
    LibTmuxException,
    OSError,
    subprocess.CalledProcessError,
)

_EMDASH_DISCOVERY_TTL = 10.0  # seconds — cache emdash session discovery


@dataclass
class PaneInfo:
    """Information about a single tmux pane within a window."""

    pane_id: str  # Stable global ID, e.g. "%3"
    index: int  # 0-based position in window
    active: bool
    command: str  # Foreground process, e.g. "claude", "bash"
    path: str  # Working directory
    width: int
    height: int


@dataclass
class TmuxWindow:
    """Information about a tmux window."""

    window_id: str
    window_name: str
    cwd: str  # Current working directory
    pane_current_command: str = ""  # Process running in active pane


class TmuxManager:
    """Manages tmux windows for Claude Code sessions."""

    def __init__(self, session_name: str | None = None):
        """Initialize tmux manager.

        Args:
            session_name: Name of the tmux session to use (default from config)
        """
        self.session_name = session_name or config.tmux_session_name
        self._server: libtmux.Server | None = None
        self._emdash_cache: list[TmuxWindow] = []
        self._emdash_cache_expires: float = 0.0

    @property
    def server(self) -> libtmux.Server:
        """Get or create tmux server connection."""
        if self._server is None:
            self._server = libtmux.Server()
        return self._server

    def _reset_server(self) -> None:
        """Reset cached server connection (e.g. after tmux server restart)."""
        self._server = None

    def get_session(self) -> libtmux.Session | None:
        """Get the tmux session if it exists."""
        try:
            return self.server.sessions.get(
                session_name=self.session_name, default=None
            )
        except _TmuxError:
            self._reset_server()
            return None

    def get_or_create_session(self) -> libtmux.Session:
        """Get existing session or create a new one."""
        session = self.get_session()
        if session:
            return session

        # Create new session with main window named specifically
        session = self.server.new_session(
            session_name=self.session_name,
            start_directory=str(Path.home()),
        )
        # Rename the default window to the main window name
        if session.windows:
            session.windows[0].rename_window(config.tmux_main_window_name)
        return session

    async def list_windows(self) -> list[TmuxWindow]:
        """List all windows in the session with their working directories.

        Returns:
            List of TmuxWindow with window info and cwd
        """

        def _sync_list_windows() -> list[TmuxWindow]:
            windows = []
            session = self.get_session()

            if not session:
                return windows

            for window in session.windows:
                name = window.window_name or ""
                window_id = window.window_id or ""
                # Skip the main window (placeholder window)
                if name == config.tmux_main_window_name:
                    continue
                # Skip our own window (auto-detect mode)
                if config.own_window_id and window_id == config.own_window_id:
                    continue

                try:
                    # Get the active pane's current path and command
                    pane = window.active_pane
                    if pane:
                        cwd = pane.pane_current_path or ""
                        pane_cmd = pane.pane_current_command or ""
                    else:
                        cwd = ""
                        pane_cmd = ""

                    windows.append(
                        TmuxWindow(
                            window_id=window.window_id or "",
                            window_name=name,
                            cwd=cwd,
                            pane_current_command=pane_cmd,
                        )
                    )
                except _TmuxError as e:
                    logger.debug("Error getting window info: %s", e)

            return windows

        return await asyncio.to_thread(_sync_list_windows)

    async def find_window_by_name(self, window_name: str) -> TmuxWindow | None:
        """Find a window by its name.

        Args:
            window_name: The window name to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_name == window_name:
                return window
        return None

    async def find_window_by_id(self, window_id: str) -> TmuxWindow | None:
        """Find a window by its tmux window ID (e.g. '@0', '@12').

        Supports foreign windows (e.g. 'emdash-claude-main-xxx:@0') by
        querying the foreign tmux session directly.

        Args:
            window_id: The tmux window ID to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        if is_foreign_window(window_id):
            return await self._find_foreign_window(window_id)
        windows = await self.list_windows()
        for window in windows:
            if window.window_id == window_id:
                return window
        return None

    async def _find_foreign_window(self, qualified_id: str) -> TmuxWindow | None:
        """Check if a foreign tmux window exists and return TmuxWindow."""
        session_name, window_id_part = qualified_id.rsplit(":", 1)
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "list-windows",
                "-t",
                session_name,
                "-F",
                "#{window_id}\t#{pane_current_path}\t#{pane_current_command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, _ = await proc.communicate()
        except TimeoutError, OSError:
            return None
        if proc.returncode != 0:
            return None
        for line in stdout.decode().strip().split("\n"):
            parts = line.split("\t", 2)
            if parts and parts[0] == window_id_part:
                cwd = parts[1] if len(parts) > 1 else ""
                cmd = parts[2] if len(parts) > 2 else ""  # noqa: PLR2004
                return TmuxWindow(
                    window_id=qualified_id,
                    window_name=session_name.removeprefix(_EMDASH_PREFIX),
                    cwd=cwd,
                    pane_current_command=cmd,
                )
        return None

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the visible text content of a window's active pane.

        Args:
            window_id: The window ID to capture
            with_ansi: If True, capture with ANSI color codes

        Returns:
            The captured text (stripped of trailing whitespace),
            or None on failure or empty content.
        """
        if with_ansi:
            return await self._capture_pane_ansi(window_id)

        return await self._capture_pane_plain(window_id)

    async def capture_pane_raw(self, window_id: str) -> tuple[str, int, int] | None:
        """Capture pane text with ANSI escapes and pane dimensions.

        Returns (raw_text, columns, rows) or None on failure. The raw text
        includes ANSI escape sequences suitable for feeding into pyte.
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            # Get dimensions and capture in one shell command
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "display-message",
                "-p",
                "-t",
                window_id,
                "#{pane_width}:#{pane_height}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    "Failed to get pane dimensions %s: %s",
                    window_id,
                    stderr.decode("utf-8", errors="replace"),
                )
                return None
            dims = stdout.decode("utf-8", errors="replace").strip()
            try:
                cols_str, rows_str = dims.split(":")
                columns, rows = int(cols_str), int(rows_str)
            except ValueError:
                return None

            # Capture with ANSI escapes
            text = await self._capture_pane_ansi(window_id)
            if text is None:
                return None
            return (text, columns, rows)
        except TimeoutError:
            logger.warning("Capture pane raw %s timed out", window_id)
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            return None
        except OSError:
            logger.exception("Unexpected error capturing pane raw %s", window_id)
            return None

    async def _capture_pane_ansi(self, window_id: str) -> str | None:
        """Capture pane with ANSI colors via tmux subprocess."""
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "capture-pane",
                "-e",
                "-p",
                "-t",
                window_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    "Failed to capture pane %s: %s",
                    window_id,
                    stderr.decode("utf-8", errors="replace"),
                )
                return None
            text = stdout.decode("utf-8", errors="replace").rstrip()
            return text if text else None
        except TimeoutError:
            logger.warning("Capture pane %s timed out", window_id)
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            return None
        except OSError:
            logger.exception("Unexpected error capturing pane %s", window_id)
            return None

    async def get_pane_title(self, window_id: str) -> str:
        """Get the terminal title of a window's active pane.

        Some CLIs (e.g. Gemini) broadcast state via OSC escape sequences
        that set the terminal title. Returns empty string on failure.
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "display-message",
                "-p",
                "-t",
                window_id,
                "#{pane_title}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return ""
            return stdout.decode("utf-8", errors="replace").strip()
        except TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            return ""
        except OSError:
            return ""

    async def _capture_pane_plain(self, window_id: str) -> str | None:
        """Capture pane as plain text via libtmux.

        Foreign windows (emdash) are captured via subprocess instead.
        """
        if is_foreign_window(window_id):
            return await self._capture_pane_ansi(window_id)

        def _sync_capture() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id, default=None)
                if not window:
                    return None
                pane = window.active_pane
                if not pane:
                    return None
                lines = pane.capture_pane()
                text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                text = text.rstrip()
                return text if text else None
            except _TmuxError as e:
                logger.warning("Failed to capture pane %s: %s", window_id, e)
                self._reset_server()
                return None

        return await asyncio.to_thread(_sync_capture)

    def _pane_send(
        self, window_id: str, chars: str, *, enter: bool, literal: bool
    ) -> bool:
        """Synchronous helper: send keys to the active pane of a window.

        Foreign windows (emdash) are handled via tmux subprocess.
        """
        if is_foreign_window(window_id):
            return self._pane_send_subprocess(
                window_id, chars, enter=enter, literal=literal
            )
        session = self.get_session()
        if not session:
            logger.warning("No tmux session found")
            return False
        try:
            window = session.windows.get(window_id=window_id, default=None)
            if not window:
                logger.warning("Window %s not found", window_id)
                return False
            pane = window.active_pane
            if not pane:
                logger.warning("No active pane in window %s", window_id)
                return False
            pane.send_keys(chars, enter=enter, literal=literal)
            return True
        except _TmuxError:
            logger.exception("Failed to send keys to window %s", window_id)
            return False

    def _pane_send_subprocess(
        self, target: str, chars: str, *, enter: bool, literal: bool
    ) -> bool:
        """Send keys via tmux subprocess (for foreign sessions)."""
        try:
            cmd = ["tmux", "send-keys", "-t", target]
            if literal:
                cmd.append("-l")
            cmd.append(chars)
            subprocess.run(cmd, timeout=5, check=False)
            if enter:
                subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    timeout=5,
                    check=False,
                )
            return True
        except subprocess.TimeoutExpired, OSError:
            logger.exception("Failed to send keys to foreign window %s", target)
            return False

    async def _ensure_vim_insert_mode(self, window_id: str) -> None:
        """Detect vim NORMAL mode and auto-enter INSERT before sending text.

        Uses a per-window cache (_vim_state) to minimize overhead:
        - False (vim off): returns immediately, zero cost.
        - True (vim on): captures pane to check INSERT indicator.
          If missing (NORMAL mode), sends ``i`` to enter INSERT.
        - Missing (unknown): probes once to determine vim state.
        """
        cached = _vim_state.get(window_id)

        # Fast path: vim is definitely off
        if cached is False:
            return

        # Check current pane for INSERT indicator
        pane_text = await self.capture_pane(window_id)
        if not pane_text:
            return

        if _has_insert_indicator(pane_text):
            _vim_state[window_id] = True
            return

        # No INSERT indicator visible.
        # If cache is None (unknown), we need to probe.
        # If cache is True (was vim), INSERT disappeared → likely NORMAL mode.
        # Both cases: send `i` and check result.
        if not await asyncio.to_thread(
            self._pane_send, window_id, "i", enter=False, literal=True
        ):
            return

        await asyncio.sleep(_VIM_PROBE_DELAY)

        pane_text = await self.capture_pane(window_id)
        if not pane_text:
            # Transient capture failure — leave state unchanged, don't backspace
            return

        if _has_insert_indicator(pane_text):
            # Vim is on — we just entered INSERT mode
            _vim_state[window_id] = True
            return

        # No INSERT indicator → vim is off (or was turned off)
        _vim_state[window_id] = False
        # Clean up the stray 'i' we typed
        await asyncio.to_thread(
            self._pane_send, window_id, "BSpace", enter=False, literal=False
        )

    async def _send_literal_then_enter(self, window_id: str, text: str) -> bool:
        """Send literal text followed by Enter with a delay.

        Claude Code's TUI sometimes interprets a rapid-fire Enter
        (arriving in the same input batch as the text) as a newline
        rather than submit.  A 500ms gap lets the TUI process the
        text before receiving Enter.

        Auto-detects vim NORMAL mode and enters INSERT before sending.
        Serialized per-window via _vim_locks to prevent interleaved probes.

        Handles ``!`` command mode: sends ``!`` first so the TUI switches
        to bash mode, waits 1s, then sends the rest.
        """
        lock = _vim_locks.setdefault(window_id, asyncio.Lock())
        async with lock:
            return await self._send_literal_then_enter_locked(window_id, text)

    async def _send_literal_then_enter_locked(self, window_id: str, text: str) -> bool:
        """Inner send implementation (must be called under per-window lock)."""
        await self._ensure_vim_insert_mode(window_id)

        if text.startswith("!"):
            if not await asyncio.to_thread(
                self._pane_send, window_id, "!", enter=False, literal=True
            ):
                return False
            rest = text[1:]
            if rest:
                await asyncio.sleep(1.0)
                if not await asyncio.to_thread(
                    self._pane_send, window_id, rest, enter=False, literal=True
                ):
                    return False
        else:
            if not await asyncio.to_thread(
                self._pane_send, window_id, text, enter=False, literal=True
            ):
                return False
        await asyncio.sleep(0.5)
        return await asyncio.to_thread(
            self._pane_send, window_id, "", enter=True, literal=False
        )

    async def send_keys(
        self, window_id: str, text: str, enter: bool = True, literal: bool = True
    ) -> bool:
        """Send keys to a specific window.

        Args:
            window_id: The window ID to send to
            text: Text to send
            enter: Whether to press enter after the text
            literal: If True, send text literally. If False, interpret special keys
                     like "Up", "Down", "Left", "Right", "Escape", "Enter".

        Returns:
            True if successful, False otherwise
        """
        if literal and enter:
            return await self._send_literal_then_enter(window_id, text)

        return await asyncio.to_thread(
            self._pane_send, window_id, text, enter=enter, literal=literal
        )

    async def kill_window(self, window_id: str) -> bool:
        """Kill a tmux window by its ID.

        Foreign windows (emdash) are never killed — they are owned externally.
        """
        if is_foreign_window(window_id):
            logger.info("Skipping kill for external window %s", window_id)
            return False

        def _sync_kill() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id, default=None)
                if not window:
                    return False
                window.kill()
                logger.info("Killed window %s", window_id)
                return True
            except _TmuxError:
                logger.exception("Failed to kill window %s", window_id)
                return False

        return await asyncio.to_thread(_sync_kill)

    async def discover_emdash_sessions(self) -> list[TmuxWindow]:
        """Discover emdash tmux sessions and return as TmuxWindow list.

        Each emdash task runs in its own tmux session named
        "emdash-{provider}-{kind}-{id}". Returns one TmuxWindow per session
        with qualified window_id (e.g. "emdash-claude-main-abc123:@0").

        Results are cached for _EMDASH_DISCOVERY_TTL seconds to avoid
        spawning N+1 subprocesses on every 1-second poll cycle.
        """
        now = asyncio.get_event_loop().time()
        if now < self._emdash_cache_expires:
            return list(self._emdash_cache)

        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "list-sessions",
                "-F",
                "#{session_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, _ = await proc.communicate()
        except TimeoutError, OSError:
            return []
        if proc.returncode != 0:
            return []

        results: list[TmuxWindow] = []
        for session_name in stdout.decode().strip().split("\n"):
            if not session_name.startswith(_EMDASH_PREFIX):
                continue
            w = await self._find_foreign_window(f"{session_name}:@0")
            if w:
                results.append(w)

        self._emdash_cache = results
        self._emdash_cache_expires = now + _EMDASH_DISCOVERY_TTL
        return list(results)

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a tmux window by its ID. Returns True on success."""

        def _sync_rename() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id, default=None)
                if not window:
                    return False
                window.rename_window(new_name)
                logger.info("Renamed window %s to %r", window_id, new_name)
                return True
            except _TmuxError:
                logger.exception("Failed to rename window %s", window_id)
                return False

        return await asyncio.to_thread(_sync_rename)

    # ── Pane-level operations ──────────────────────────────────────────

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """List all panes in a window.

        Returns an empty list if the window is not found or on error.
        """

        def _sync_list_panes() -> list[PaneInfo]:
            session = self.get_session()
            if not session:
                return []
            try:
                window = session.windows.get(window_id=window_id, default=None)
                if not window:
                    return []
                result: list[PaneInfo] = []
                for pane in window.panes:
                    result.append(
                        PaneInfo(
                            pane_id=pane.pane_id or "",
                            index=int(pane.pane_index or 0),
                            active=pane.pane_active == "1",
                            command=pane.pane_current_command or "",
                            path=pane.pane_current_path or "",
                            width=int(pane.pane_width or 0),
                            height=int(pane.pane_height or 0),
                        )
                    )
                return result
            except _TmuxError as exc:
                logger.warning("Failed to list panes for %s: %s", window_id, exc)
                self._reset_server()
                return []

        return await asyncio.to_thread(_sync_list_panes)

    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,
    ) -> str | None:
        """Capture visible text of a specific pane (by stable pane ID like '%3').

        Unlike capture_pane() which targets the active pane of a window,
        this targets a specific pane regardless of whether it is active.

        When window_id is given, the pane must belong to that window (prevents
        cross-window access via crafted pane IDs).
        """
        if with_ansi:
            if window_id:
                # Validate pane belongs to the specified window before capture
                panes = await self.list_panes(window_id)
                if not any(p.pane_id == pane_id for p in panes):
                    logger.warning("Pane %s not found in window %s", pane_id, window_id)
                    return None
            return await self._capture_pane_ansi(pane_id)

        def _sync_capture() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                pane = self._find_pane(pane_id, session, window_id=window_id)
                if not pane:
                    return None
                lines = pane.capture_pane()
                text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                text = text.rstrip()
                return text if text else None
            except _TmuxError as exc:
                logger.warning("Failed to capture pane %s: %s", pane_id, exc)
                self._reset_server()
                return None

        return await asyncio.to_thread(_sync_capture)

    async def send_keys_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Send keys to a specific pane (by stable pane ID like '%3').

        Unlike send_keys() which targets the active pane of a window,
        this targets a specific pane regardless of whether it is active.

        When window_id is given, the pane must belong to that window (prevents
        cross-window access via crafted pane IDs).
        """

        def _sync_send() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                pane = self._find_pane(pane_id, session, window_id=window_id)
                if not pane:
                    logger.warning("Pane %s not found", pane_id)
                    return False
                pane.send_keys(text, enter=enter, literal=literal)
                return True
            except _TmuxError:
                logger.exception("Failed to send keys to pane %s", pane_id)
                return False

        return await asyncio.to_thread(_sync_send)

    @staticmethod
    def _find_pane(
        pane_id: str,
        session: libtmux.Session,
        *,
        window_id: str | None = None,
    ) -> libtmux.Pane | None:
        """Find a pane by its stable ID (e.g. '%3').

        When window_id is given, only searches that window's panes (prevents
        cross-window access). Otherwise searches all windows in the session.
        """
        windows = session.windows
        if window_id:
            windows = [w for w in windows if w.window_id == window_id]
        for window in windows:
            for pane in window.panes:
                if pane.pane_id == pane_id:
                    return pane
        return None

    async def create_window(
        self,
        work_dir: str,
        window_name: str | None = None,
        start_agent: bool = True,
        agent_args: str = "",
        launch_command: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Create a new tmux window and optionally start an agent CLI.

        Args:
            work_dir: Working directory for the new window
            window_name: Optional window name (defaults to directory name)
            start_agent: Whether to start the agent CLI command
            agent_args: Extra arguments appended to the launch command
                        (e.g. "--continue", "--resume <id>")
            launch_command: The CLI command to run (e.g. "claude", "codex", "gemini")

        Returns:
            Tuple of (success, message, window_name, window_id)
        """
        # Validate directory first
        path = Path(work_dir).expanduser().resolve()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", "", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", "", ""

        # Create window name, adding suffix if name already exists
        final_window_name = window_name if window_name else path.name

        # Check for existing window name
        base_name = final_window_name
        counter = 2
        while await self.find_window_by_name(final_window_name):
            final_window_name = f"{base_name}-{counter}"
            counter += 1

        # Create window in thread
        def _create_and_start() -> tuple[bool, str, str, str]:
            session = self.get_or_create_session()
            try:
                # Create new window
                window = session.new_window(
                    window_name=final_window_name,
                    start_directory=str(path),
                )

                new_window_id = window.window_id or ""

                # Start agent CLI if requested
                if start_agent and launch_command:
                    pane = window.active_pane
                    if pane:
                        cmd = launch_command
                        if agent_args:
                            cmd = f"{cmd} {agent_args}"
                        pane.send_keys(cmd, enter=True)

                logger.info(
                    "Created window '%s' (id=%s) at %s",
                    final_window_name,
                    new_window_id,
                    path,
                )
                return (
                    True,
                    f"Created window '{final_window_name}' at {path}",
                    final_window_name,
                    new_window_id,
                )

            except _TmuxError as e:
                logger.exception("Failed to create window")
                return False, f"Failed to create window: {e}", "", ""

        return await asyncio.to_thread(_create_and_start)


# Global instance with default session name
tmux_manager = TmuxManager()
