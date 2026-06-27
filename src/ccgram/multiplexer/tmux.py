"""Tmux backend for the Multiplexer contract, via libtmux.

The first concrete backend behind ``multiplexer.base.Multiplexer``.  Wraps
libtmux to provide async-friendly operations on a single tmux session:
  - list_windows / find_window_by_name: discover Claude Code windows.
  - capture_pane: read terminal content (plain or with ANSI colors).
  - send_keys: forward user input or control keys to a window.
  - create_window / kill_window: lifecycle management.
  - list_panes / capture_pane_by_id / send_keys_to_pane: pane-level ops.
  - Vim mode detection: auto-enter INSERT mode before sending text when
    Claude Code's /vim mode is active and the TUI is in NORMAL mode.

All blocking libtmux calls are wrapped in asyncio.to_thread().

``TmuxManager`` satisfies the ``Multiplexer`` Protocol (neutral value types +
``capabilities``).  The Protocol method names (``find_window``, ``capture``,
``send``, ``set_title``, ``foreground``, …) and the legacy names
(``find_window_by_id``, ``capture_pane``, ``send_keys``, ``stamp_pane_title``,
…) coexist during call-site migration; the legacy names are removed in Task 4.

Key class: TmuxManager (singleton instantiated as `tmux_manager`).
Module-level: _vim_state cache, _vim_locks for per-window send serialization.
"""

from __future__ import annotations

import asyncio
import contextlib
import structlog
import subprocess
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import libtmux
from libtmux.exc import LibTmuxException

from ..config import config
from .base import (
    AgentStatus,
    CaptureResult,
    ForegroundInfo,
    MultiplexerCapabilities,
    MuxEvent,
    PaneDims,
    PaneInfo,
    WindowRef,
    WorkspaceRef,
)
from .vim_state import (
    _vim_locks,
    _vim_state,
    clear_vim_state,
    has_insert_indicator,
    notify_vim_insert_seen,
    reset_vim_state,
)

__all__ = [
    "PaneInfo",
    "TmuxManager",
    "TmuxWindow",
    "_vim_locks",
    "_vim_state",
    "clear_vim_state",
    "has_insert_indicator",
    "notify_vim_insert_seen",
    "reset_vim_state",
    "tmux_manager",
]

logger = structlog.get_logger()

# Backward-compat alias: callers still import ``TmuxWindow`` from this module.
# ``WindowRef`` is field-compatible (Task 1), so the alias is exact.
TmuxWindow = WindowRef

# Vim-insert detection state moved to ``multiplexer.vim_state`` (backend-neutral)
# so the polling layer can import the helpers without importing this backend
# (F1 boundary).  ``send`` below reads ``_vim_state`` / ``_vim_locks`` and the
# ``has_insert_indicator`` probe imported above.


_TmuxError = (
    LibTmuxException,
    OSError,
    subprocess.CalledProcessError,
)

# Static capability declaration for the tmux backend (design Task 2).
_TMUX_CAPABILITIES = MultiplexerCapabilities(
    name="tmux",
    ids_stable_across_restart=True,
    exposes_pane_tty=True,
    native_agent_status=False,
    read_max_lines=None,
    self_identify_env="TMUX_PANE",
    supports_event_stream=False,
    native_worktrees=False,
)


class TmuxManager:
    """Manages tmux windows for Claude Code sessions.

    Satisfies the ``Multiplexer`` Protocol: returns the neutral value types
    (``WindowRef``/``PaneInfo``/``CaptureResult``/``ForegroundInfo``/
    ``PaneDims``) and exposes ``capabilities``.
    """

    @property
    def capabilities(self) -> MultiplexerCapabilities:
        """Return the static capability declaration for the tmux backend."""
        return _TMUX_CAPABILITIES

    def __init__(self, session_name: str | None = None):
        """Initialize tmux manager.

        Args:
            session_name: Name of the tmux session to use (default from config)
        """
        self.session_name = session_name or config.tmux_session_name
        self._server: libtmux.Server | None = None

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
                # Skip hidden windows (name starts with underscore)
                if name.startswith("_"):
                    continue

                try:
                    # Get the active pane's current path, command, and dimensions
                    pane = window.active_pane
                    if pane:
                        cwd = pane.pane_current_path or ""
                        pane_cmd = pane.pane_current_command or ""
                        pane_tty = getattr(pane, "pane_tty", "") or ""
                        pw = int(pane.pane_width or 0)
                        ph = int(pane.pane_height or 0)
                    else:
                        cwd = ""
                        pane_cmd = ""
                        pane_tty = ""
                        pw = 0
                        ph = 0

                    windows.append(
                        TmuxWindow(
                            window_id=window.window_id or "",
                            window_name=name,
                            cwd=cwd,
                            pane_current_command=pane_cmd,
                            pane_tty=pane_tty,
                            pane_width=pw,
                            pane_height=ph,
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

        Args:
            window_id: The tmux window ID to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_id == window_id:
                return window
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

    async def capture_pane_scrollback(
        self, window_id: str, history: int = 200
    ) -> str | None:
        """Capture pane text including scrollback history (plain text, no ANSI).

        Uses ``tmux capture-pane -p -J -S -{history}``. The ``-J`` flag joins
        wrapped lines so prompt markers are never split across lines on narrow
        terminals. Returns stripped text or None on failure.
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            cmd = [
                "tmux",
                "capture-pane",
                "-p",
                "-J",
                "-S",
                f"-{history}",
                "-t",
                window_id,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5.0):
                stdout, _ = await proc.communicate()
            text = stdout.decode("utf-8", errors="replace").rstrip()
            return text if text else None
        except TimeoutError:
            if proc:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
            logger.debug("capture_pane_scrollback timed out", window_id=window_id)
            return None
        except OSError as exc:
            logger.debug(
                "capture_pane_scrollback failed", window_id=window_id, error=str(exc)
            )
            return None

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
                # Window/pane gone mid-capture — a race in the live-view poll,
                # not an operator-actionable error. TimeoutError below stays
                # WARNING (tmux genuinely wedged).
                logger.debug(
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
                # Window/pane gone mid-capture — live-view race, not actionable.
                logger.debug(
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

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """Set pane title to ``ccgram:<provider>`` for instant re-detection.

        Uses ``tmux select-pane -T`` to set the title directly, avoiding
        ``send_keys`` which would deliver the command as input to agent CLIs.
        """
        title = f"ccgram:{provider_name}"
        target = f"{self.session_name}:{window_id}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "select-pane",
                "-t",
                target,
                "-T",
                title,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except OSError:
            pass

    async def _capture_pane_plain(self, window_id: str) -> str | None:
        """Capture pane as plain text via libtmux."""

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
                if "can't find window" in str(e):
                    # Window closed mid-capture — race, not an error.
                    logger.debug("Pane capture skipped (window gone): %s", window_id)
                else:
                    # Other libtmux errors here are transient races on the
                    # status-poll path (~1s); server reset below recovers.
                    logger.debug("Failed to capture pane %s: %s", window_id, e)
                self._reset_server()
                return None

        return await asyncio.to_thread(_sync_capture)

    def _pane_send(
        self, window_id: str, chars: str, *, enter: bool, literal: bool
    ) -> bool:
        """Synchronous helper: send keys to the active pane of a window."""
        session = self.get_session()
        if not session:
            logger.debug("No tmux session found")
            return False
        try:
            window = session.windows.get(window_id=window_id, default=None)
            if not window:
                logger.debug("Window %s not found", window_id)
                return False
            pane = window.active_pane
            if not pane:
                logger.debug("No active pane in window %s", window_id)
                return False
            pane.send_keys(chars, enter=enter, literal=literal)
            return True
        except _TmuxError:
            logger.exception("Failed to send keys to window %s", window_id)
            return False

    async def _ensure_vim_insert_mode(self, window_id: str) -> None:
        """Enter vim INSERT mode before sending text — only when vim is known on.

        Vim state is observed by status polling: notify_vim_insert_seen sets the
        cache to True when it renders ``-- INSERT --``. We act only on a
        positively-confirmed vim window:
        - True (vim on): capture the pane; if INSERT is not showing we are in
          NORMAL mode, so send ``i`` to enter INSERT.
        - False / unknown (None): no-op. We never speculatively type ``i`` here.
          Doing so leaked a literal ``i`` into the message whenever the pane was
          not actually in vim NORMAL mode — the common case, since Claude Code's
          default prompt is not vim. The old probe could not tell ``i entered
          INSERT`` from ``i typed while already in INSERT``, so it leaked the
          probe key on capture failures and on stale captures.

        Resilient by construction: a capture failure or an already-INSERT pane
        sends nothing, so no key can leak. The only ``i`` ever sent is into a
        window polling has confirmed is in vim and that currently shows no
        INSERT indicator — i.e. genuine NORMAL mode.

        Residual caveat: if a confirmed-vim window leaves vim mode, the cache
        stays True until the window is cleaned up, so the first post-exit send
        may still emit a stray ``i``. Far narrower than the previous
        every-message leak and only affects users who toggle vim off mid-window.
        """
        if _vim_state.get(window_id) is not True:
            return

        pane_text = await self.capture_pane(window_id)
        if not pane_text:
            return

        if has_insert_indicator(pane_text):
            return  # already in INSERT

        # vim on, no INSERT indicator → NORMAL mode; enter INSERT.
        await asyncio.to_thread(
            self._pane_send, window_id, "i", enter=False, literal=True
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
        self,
        window_id: str,
        text: str,
        enter: bool = True,
        literal: bool = True,
        *,
        raw: bool = False,
    ) -> bool:
        """Send keys to a specific window.

        Args:
            window_id: The window ID to send to
            text: Text to send
            enter: Whether to press enter after the text
            literal: If True, send text literally. If False, interpret special keys
                     like "Up", "Down", "Left", "Right", "Escape", "Enter".
            raw: If True, bypass TUI-specific workarounds (``!`` prefix splitting,
                 vim mode detection, Enter delay). Use for plain shell windows.

        Returns:
            True if successful, False otherwise
        """
        if literal and enter and not raw:
            return await self._send_literal_then_enter(window_id, text)

        return await asyncio.to_thread(
            self._pane_send, window_id, text, enter=enter, literal=literal
        )

    async def kill_window(self, window_id: str) -> bool:
        """Kill a tmux window by its ID."""

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
                # Miniapp polls panes at ~0.2s; a gone window races here. Debug,
                # not warning — server reset below recovers.
                logger.debug("Failed to list panes for %s: %s", window_id, exc)
                self._reset_server()
                return []

        return await asyncio.to_thread(_sync_list_panes)

    async def split_window(self, window_id: str) -> str | None:
        """Split a window's active pane via libtmux; return the new pane id.

        Returns None when the window is gone or the split fails.
        """

        def _sync_split() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id, default=None)
                if not window:
                    return None
                pane = window.split()
                return pane.pane_id or None
            except _TmuxError as exc:
                logger.debug("Failed to split window %s: %s", window_id, exc)
                self._reset_server()
                return None

        return await asyncio.to_thread(_sync_split)

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
                    logger.debug("Pane %s not found in window %s", pane_id, window_id)
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
                # Miniapp polls captures at ~0.2s; a gone pane races here. Debug,
                # not warning — server reset below recovers.
                logger.debug("Failed to capture pane %s: %s", pane_id, exc)
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
                    logger.debug("Pane %s not found", pane_id)
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

    @staticmethod
    def _start_agent_in_pane(
        pane: libtmux.Pane,
        launch_command: str,
        agent_args: str,
    ) -> None:
        """Send launch command to pane, appending agent_args if provided."""
        cmd = launch_command
        if agent_args:
            cmd = f"{cmd} {agent_args}"
        pane.send_keys(cmd, enter=True, literal=True)

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """tmux has no workspace concept — always returns ``[]``."""
        return []

    async def create_window(
        self,
        work_dir: str,
        window_name: str | None = None,
        start_agent: bool = True,
        agent_args: str = "",
        launch_command: str | None = None,
        *,
        workspace_id: str | None = None,  # noqa: ARG002 — tmux has no workspaces
    ) -> tuple[bool, str, str, str]:
        """Create a new tmux window and optionally start an agent CLI.

        Args:
            work_dir: Working directory for the new window
            window_name: Optional window name (defaults to directory name)
            start_agent: Whether to start the agent CLI command
            agent_args: Extra arguments appended to the launch command
                        (e.g. "--continue", "--resume <id>")
            launch_command: The CLI command to run (e.g. "claude", "codex", "gemini")
            workspace_id: Ignored — tmux has no workspace concept.

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
                pane = window.active_pane

                # Disable interactive editors — Telegram users can't see
                # tmux popups or terminal overlays opened by plugins
                if pane and new_window_id:
                    pane.send_keys(
                        "export EDITOR=true VISUAL=true",
                        enter=True,
                    )

                if not (start_agent and launch_command):
                    window.set_option("automatic-rename", "off")
                elif pane:
                    self._start_agent_in_pane(pane, launch_command, agent_args)

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

    # ── Multiplexer Protocol surface ───────────────────────────────────
    # Neutral-typed wrappers over the libtmux-specific methods above.
    # Call sites migrate to these names in Task 4; the legacy names are
    # then removed.  Behavior is identical — these only rename/repackage.

    async def ensure_session(self) -> None:
        """Ensure the tmux session exists (``get_or_create_session``)."""
        await asyncio.to_thread(self.get_or_create_session)

    async def find_window(self, window_id: str) -> WindowRef | None:
        """Find a window by its opaque ID string (alias of ``find_window_by_id``)."""
        return await self.find_window_by_id(window_id)

    async def capture(
        self, window_id: str, *, ansi: bool = False
    ) -> CaptureResult | None:
        """Capture the visible text of the active pane as a ``CaptureResult``."""
        text = await self.capture_pane(window_id, with_ansi=ansi)
        if text is None:
            return None
        return CaptureResult(text=text)

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Capture pane text with scrollback, clamped to ``read_max_lines``.

        tmux has no line cap (``read_max_lines is None``), so ``truncated`` is
        always False here; the clamp exists for backends that cap (herdr).
        """
        max_lines = self.capabilities.read_max_lines
        effective = lines
        truncated = False
        if max_lines is not None and lines > max_lines:
            effective = max_lines
            truncated = True
        text = await self.capture_pane_scrollback(window_id, history=effective)
        if text is None:
            return None
        return CaptureResult(text=text, truncated=truncated)

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Return the active pane's column/row dimensions, or None on failure."""
        proc: asyncio.subprocess.Process | None = None
        try:
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
                stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            dims = stdout.decode("utf-8", errors="replace").strip()
            cols_str, rows_str = dims.split(":")
            return PaneDims(width=int(cols_str), height=int(rows_str))
        except TimeoutError, ValueError:
            if proc:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
            return None
        except OSError:
            return None

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,
    ) -> bool:
        """Send text to the active pane (alias of ``send_keys``)."""
        return await self.send_keys(
            window_id, text, enter=enter, literal=literal, raw=raw
        )

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Send text to a specific pane (alias of ``send_keys_to_pane``)."""
        return await self.send_keys_to_pane(
            pane_id, text, enter=enter, literal=literal, window_id=window_id
        )

    async def create_worktree_window(
        self,
        repo_path: str,  # noqa: ARG002 — tmux has no native worktrees
        worktree_path: str,  # noqa: ARG002
        branch: str,  # noqa: ARG002
        *,
        window_name: str | None = None,  # noqa: ARG002
        launch_command: str | None = None,  # noqa: ARG002
    ) -> tuple[bool, str, str, str]:
        """Not supported on tmux (``native_worktrees`` is False).

        Callers gate on ``capabilities.native_worktrees`` and use ccgram's own
        ``git worktree add`` + ``create_window`` path instead, so this is never
        reached. Returns a failure tuple for safety rather than raising.
        """
        return False, "worktree delegation unsupported on tmux", "", ""

    async def watch_events(
        self,
        window_ids: Sequence[str],  # noqa: ARG002 — tmux has no event stream
    ) -> AsyncGenerator[MuxEvent, None]:
        """No event stream on tmux (``supports_event_stream`` is False).

        Yields nothing; consumers gate on ``capabilities.supports_event_stream``
        and never call this. The empty ``for`` loop makes it an async generator.
        """
        for _ in ():  # pragma: no cover
            yield MuxEvent(kind="", window_id="")

    async def set_title(self, window_id: str, provider_name: str) -> None:
        """Set the pane title for re-detection (alias of ``stamp_pane_title``)."""
        await self.stamp_pane_title(window_id, provider_name)

    async def agent_status(
        self,
        window_id: str,  # noqa: ARG002 — protocol signature
    ) -> AgentStatus | None:
        """tmux has no native agent status (``native_agent_status`` is False).

        Returns None so callers fall back to terminal scraping.
        """
        return None

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """Return foreground process info for the active pane via ``ps -t``.

        Reads the window's ``pane_tty`` then the foreground process group
        leader from ``ps``.  ``cwd`` comes from the window's reported cwd.
        Returns None when the window is gone or no foreground process exists.
        """
        window = await self.find_window_by_id(window_id)
        if not window or not window.pane_tty:
            return None
        parsed = await self._ps_foreground(window.pane_tty)
        if parsed is None:
            return None
        pid, pgid, argv = parsed
        return ForegroundInfo(
            pid=pid,
            pgid=pgid,
            argv=argv,
            cwd=window.cwd,
            tty=window.pane_tty,
        )

    @staticmethod
    async def _ps_foreground(tty_path: str) -> tuple[int, int, list[str]] | None:
        """Parse ``ps -t <tty>`` for the foreground group leader.

        Returns ``(pid, pgid, argv)`` for the group leader (pid == pgid) among
        foreground processes (``+`` in stat), or the first foreground process
        as a fallback.  None on any error or no foreground process.
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "ps",
                "-t",
                tty_path,
                "-o",
                "pid=,pgid=,stat=,args=",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(3.0):
                stdout, _ = await proc.communicate()
        except TimeoutError:
            if proc:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
            return None
        except OSError:
            return None
        if proc.returncode != 0:
            return None

        fallback: tuple[int, int, list[str]] | None = None
        for line in stdout.decode("utf-8", errors="replace").strip().splitlines():
            parsed = TmuxManager._parse_ps_line(line)
            if parsed is None:
                continue
            if parsed[0] == parsed[1]:  # pid == pgid → group leader
                return parsed
            if fallback is None:
                fallback = parsed
        return fallback

    @staticmethod
    def _parse_ps_line(line: str) -> tuple[int, int, list[str]] | None:
        """Parse one ``ps`` line into ``(pid, pgid, argv)`` if foreground.

        Returns None for non-foreground lines (no ``+`` in stat) or malformed
        rows.
        """
        parts = line.split(None, 3)
        if len(parts) < 4:  # noqa: PLR2004
            return None
        pid_s, pgid_s, stat, args = parts
        if "+" not in stat:
            return None
        try:
            return int(pid_s), int(pgid_s), args.split()
        except ValueError:
            return None


# Global instance with default session name
tmux_manager = TmuxManager()

# ``send_to_window`` / ``send_followup_to_window`` moved to
# ``multiplexer.window_ops`` (backend-neutral; they route through the proxy).
