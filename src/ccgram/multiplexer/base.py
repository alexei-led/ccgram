"""Multiplexer contract — pure types, no I/O.

Defines the ``Multiplexer`` Protocol and the neutral value types that all
backends return.  No backend imports, no subprocess, no libtmux, no asyncio
subprocess — this module is dependency-free so that callers can type against it
without pulling in any backend.

Value-type field names are chosen to be field-compatible with the existing
``TmuxWindow`` / ``PaneInfo`` dataclasses in ``tmux_manager.py`` so the tmux
refactor in Task 2 is mechanical.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ── Value types ────────────────────────────────────────────────────────


@dataclass
class WindowRef:
    """Neutral representation of a multiplexer window (tmux window / herdr pane).

    Field names match the existing ``TmuxWindow`` fields so Task 2 call-site
    migration is mechanical.
    """

    window_id: str
    window_name: str
    cwd: str
    pane_current_command: str = ""
    pane_tty: str = ""
    pane_width: int = 0
    pane_height: int = 0


@dataclass
class PaneInfo:
    """Neutral representation of a pane within a window.

    Field names match the existing ``PaneInfo`` in ``tmux_manager.py``.
    """

    pane_id: str  # e.g. "%3" for tmux, "w2:p1" for herdr
    index: int
    active: bool
    command: str  # Foreground process name
    path: str  # Working directory
    width: int
    height: int


@dataclass
class CaptureResult:
    """Result of a pane capture operation."""

    text: str  # Captured text (plain or ANSI depending on the call)
    truncated: bool = (
        False  # True when scrollback was clamped (e.g. herdr 1000-line cap)
    )


@dataclass
class ForegroundInfo:
    """Foreground process info from a multiplexer pane.

    tmux backend: from ``pane_tty`` + ``ps -t <tty>``.
    herdr backend: from ``pane process-info`` → ``foreground_processes[]``.
    No tty on macOS herdr — ``tty`` is empty string in that case.
    """

    pid: int
    pgid: int
    argv: list[str]
    cwd: str
    tty: str = ""  # Empty when not available (herdr on macOS)


@dataclass
class PaneDims:
    """Terminal dimensions of a pane."""

    width: int  # Columns
    height: int  # Rows


@dataclass
class WorkspaceRef:
    """Neutral representation of a multiplexer workspace (herdr workspace).

    tmux has no workspace concept; its backend returns ``[]`` from
    ``list_workspaces``.  herdr backends return one entry per workspace.
    The ``workspace_id`` is an opaque string (herdr-internal; callers treat
    it as a token to pass back to ``create_window``).
    """

    workspace_id: str  # Opaque ID — pass to create_window to pin the workspace
    label: str  # Human-readable name
    cwd: str  # Root directory of the workspace


@dataclass(frozen=True)
class AgentStatus:
    """Native agent run-state reported by the multiplexer.

    Only backends with ``capabilities.native_agent_status`` populate this
    (herdr); others return ``None`` from ``agent_status()`` so callers fall
    back to terminal scraping. ``state`` is the backend's raw label — herdr
    reports ``working`` / ``idle`` / ``done`` / ``blocked`` / ``unknown``.
    """

    state: str
    agent: str = ""  # Agent name (e.g. "claude", "codex"); "" when none
    custom_status: str = ""  # Optional short activity label (herdr custom_status)


@dataclass(frozen=True)
class MuxEvent:
    """A push event from a backend's event stream (``supports_event_stream``).

    Backends that lack an event stream (tmux) never emit these. The backend
    resolves the event to a ``window_id`` (tab/window id) so consumers stay
    backend-neutral. ``kind``:

    - ``"agent_status"`` — the pane's native agent run-state changed; ``status``
      carries the new ``AgentStatus``.
    - ``"window_died"`` — the window's agent process exited or the window/tab
      closed (``pane.exited`` / ``tab.closed`` on herdr).
    """

    kind: str
    window_id: str
    pane_id: str = ""
    status: AgentStatus | None = None


# ── Capabilities ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class MultiplexerCapabilities:
    """Immutable capability declaration for a multiplexer backend.

    Gates UX and control flow — callers must use these flags, never
    ``caps.name == "tmux"`` conditionals.
    """

    name: str
    """Backend identifier — for logging and doctor only.  Not for conditionals."""

    ids_stable_across_restart: bool
    """True when window IDs survive a server restart (tmux: True; herdr: False)."""

    exposes_pane_tty: bool
    """True when ``foreground()`` can return a tty device path (tmux: True)."""

    native_agent_status: bool
    """True when the backend exposes agent status natively (herdr: True)."""

    read_max_lines: int | None
    """Maximum scrollback lines the backend can return; None = unlimited (tmux)."""

    self_identify_env: str
    """Environment variable set by the backend for hook identity resolution."""

    supports_event_stream: bool
    """True when the backend has a push event stream (herdr: True, tmux: False).

    Gates ``watch_events`` consumption: ``bootstrap.start_event_stream`` only
    starts the ``EventStreamMonitor`` on backends with this flag (herdr); tmux's
    ``watch_events`` is an empty async generator that is never consumed.
    """

    native_worktrees: bool
    """True when the backend creates git worktrees natively (herdr: True).

    Gates the ``/new`` worktree step onto ``create_worktree_window``: herdr makes
    the checkout and groups it under the parent repo in its workspace UI. tmux
    (False) keeps ccgram's own ``git worktree add`` + ``create_window`` path.
    """


# ── Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class Multiplexer(Protocol):
    """Contract every terminal-multiplexer backend must satisfy.

    Method surface mirrors the current ``TmuxManager`` public API, normalised
    to neutral value types.  All methods are async.

    Callers import this Protocol from ``multiplexer.base`` and receive a
    concrete instance from the ``multiplexer`` module-level proxy (wired by
    ``bootstrap.py``).  No caller should import a concrete backend
    (``multiplexer.tmux``, ``multiplexer.herdr``) directly.
    """

    @property
    def capabilities(self) -> MultiplexerCapabilities:
        """Return the static capability declaration for this backend."""
        ...

    async def ensure_session(self) -> None:
        """Ensure the multiplexer session/server is reachable.

        tmux: ``get_or_create_session()``.
        herdr: verify socket is alive and at least one workspace exists.
        """
        ...

    async def list_windows(self) -> list[WindowRef]:
        """List all agent windows in the session."""
        ...

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """List all workspaces in the session.

        tmux returns ``[]`` (no workspace concept).
        herdr returns one ``WorkspaceRef`` per workspace.
        The ``workspace_id`` is an opaque token — pass it to ``create_window``
        to pin the new tab inside an existing workspace.
        """
        ...

    async def find_window(self, window_id: str) -> WindowRef | None:
        """Find a window by its opaque ID string.

        Returns None when the window does not exist or is no longer alive.
        """
        ...

    async def capture(
        self, window_id: str, *, ansi: bool = False
    ) -> CaptureResult | None:
        """Capture the visible text of the active pane.

        Returns None on failure (window gone, timeout, socket error).
        """
        ...

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Capture pane text including scrollback history (plain text).

        ``lines`` is clamped to ``capabilities.read_max_lines`` when set.
        Returns None on failure.
        """
        ...

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Return the active pane's column/row dimensions.

        Returns None when the window is gone or the query fails.
        """
        ...

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,
    ) -> bool:
        """Send text to the active pane of a window.

        ``raw=True`` bypasses TUI-specific workarounds (vim detection, Enter
        delay, ``!``-prefix splitting).
        Returns True on success.
        """
        ...

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Send text to a specific pane (by stable pane ID).

        ``window_id`` limits the search to that window (cross-window access
        prevention).  Returns True on success.
        """
        ...

    async def kill_window(self, window_id: str) -> bool:
        """Kill/close a window.  Returns True on success."""
        ...

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a window.  Returns True on success."""
        ...

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """List all panes in a window.  Empty list on error or missing window."""
        ...

    async def create_window(
        self,
        work_dir: str,
        window_name: str | None = None,
        start_agent: bool = True,
        agent_args: str = "",
        launch_command: str | None = None,
        *,
        workspace_id: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Create a new window and optionally start an agent CLI.

        ``workspace_id`` is only meaningful on backends that have a workspace
        concept (herdr).  When provided, the new window is created inside the
        given workspace instead of resolving one from *work_dir*.  tmux ignores
        the parameter.

        Returns ``(success, message, window_name, window_id)``.
        """
        ...

    async def create_worktree_window(
        self,
        repo_path: str,
        worktree_path: str,
        branch: str,
        *,
        window_name: str | None = None,
        launch_command: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Create a git worktree *and* the window that runs in it (one step).

        Only called on backends with ``capabilities.native_worktrees`` (herdr):
        the backend makes the checkout at *worktree_path* on *branch* (off the
        repo at *repo_path*) and opens a window/workspace grouped under the
        parent repo, then launches *launch_command*. Backends without the
        capability return ``(False, "<reason>", "", "")`` — callers gate on the
        flag and never reach them.

        Returns ``(success, message, window_name, window_id)`` like
        ``create_window`` (``window_id`` is the new tab/window id).
        """
        ...

    async def set_title(self, window_id: str, provider_name: str) -> None:
        """Set the pane title for instant provider re-detection.

        tmux: ``select-pane -T ccgram:<provider>``.
        herdr: ``pane report-metadata --title``.
        """
        ...

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """Return foreground process info for the active pane.

        Uses ``pane_tty`` + ``ps -t`` on tmux; ``pane process-info`` on herdr.
        Returns None when the window is gone or no foreground process exists.
        """
        ...

    async def agent_status(self, window_id: str) -> "AgentStatus | None":
        """Return the active pane's native agent run-state, or None.

        Only meaningful when ``capabilities.native_agent_status`` is True
        (herdr reads ``pane.agent_status``). tmux has no native agent status
        and returns None, so callers fall back to terminal scraping.
        """
        ...

    async def split_window(self, window_id: str) -> str | None:
        """Split the window's active pane; return the new pane id, or None.

        Adds a sibling pane to the window/tab (the multi-pane "agent team"
        shape): herdr ``pane split``, tmux ``window.split()``. The returned id
        is a real pane id (``%N`` for tmux, ``wN:pK`` for herdr) and is
        discoverable via ``list_panes`` / the ``/panes`` command. None on
        failure (window gone, backend error).
        """
        ...

    def watch_events(self, window_ids: Sequence[str]) -> AsyncGenerator[MuxEvent, None]:
        """Yield push events for *window_ids* until the iterator is cancelled.

        Only meaningful on backends with ``capabilities.supports_event_stream``
        (herdr). herdr opens a persistent subscription — global window-death
        plus per-pane agent-status for the active panes of *window_ids* — and
        reconnects with backoff on a dropped connection, repriming agent status
        on each (re)connect. tmux returns an empty async iterator.

        The watched set is fixed for the lifetime of one call; herdr cannot add
        subscriptions to a live connection, so the consumer cancels this
        iterator and calls again with the new set when bindings change.
        """
        ...

    # ── Transitional surface ───────────────────────────────────────────
    #
    # Methods below mirror the historical ``tmux_manager`` public API that
    # callers still use directly.  They are part of the contract so callers can
    # depend only on the ``multiplexer`` proxy (typed against this Protocol)
    # without importing a concrete backend (F1) and without being rewritten to
    # the value-type surface above (design BC rule: "do not rewrite callers").
    # The value-type methods (``find_window`` / ``capture`` / ``send`` / …) are
    # the forward surface for new herdr-aware code; both backends implement
    # both.  An ``architecture-review`` may prune these once callers adopt the
    # value-type surface (Tasks 10–11).

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """Find a window by its opaque ID (legacy alias of ``find_window``)."""
        ...

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the active pane's visible text as a plain string.

        Returns the captured text (stripped) or None on failure/empty.
        """
        ...

    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,
    ) -> str | None:
        """Capture a specific pane's visible text by stable pane ID.

        ``window_id`` limits the lookup to that window (cross-window guard).
        Returns the text or None on failure.
        """
        ...

    async def capture_pane_scrollback(
        self, window_id: str, history: int = 200
    ) -> str | None:
        """Capture pane text including scrollback history (plain text).

        ``history`` is clamped to ``capabilities.read_max_lines`` when set.
        Returns the text or None on failure.
        """
        ...

    async def send_keys(
        self,
        window_id: str,
        text: str,
        enter: bool = True,
        literal: bool = True,
        *,
        raw: bool = False,
    ) -> bool:
        """Send text to a window's active pane (legacy alias of ``send``)."""
        ...

    async def send_keys_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Send text to a specific pane (legacy alias of ``send_to_pane``)."""
        ...

    async def get_pane_title(self, window_id: str) -> str:
        """Return the active pane's terminal title, or '' on failure."""
        ...

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """Set the pane title for re-detection (legacy alias of ``set_title``)."""
        ...
