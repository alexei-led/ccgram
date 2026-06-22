"""Herdr backend for the Multiplexer contract, via the herdr CLI/socket.

Anti-corruption layer over `herdr <https://github.com/ogulcancelik/herdr>`_'s
Unix-socket JSON-RPC CLI. Every herdr JSON shape (``pane_info`` / ``pane_list``
/ ``pane_process_info`` / ``pane_layout`` / ``tab_created`` …) and every
``wN:pN``/``wN:tN`` id string stays **private** to this module; callers see
only the neutral value types from ``multiplexer.base`` (design "Module map":
herdr.py is adapter, anti-corruption).

Identity mapping: herdr's ``tab_id`` (``"w2:t1"``) *is* the ``window_id``
string (tab identity — one ccgram topic = one herdr tab). A split tab (team)
is one topic with multiple panes; pane ops are resolved tab→active-pane in
Task 4.

The backend shells out to the ``herdr`` CLI (which the design explicitly allows
as an alternative to talking the socket directly); the socket path is passed
through ``$HERDR_SOCKET_PATH``. The command runner is injectable so unit tests
feed JSON fixtures without a live socket and the constructor stays I/O-free
(the proxy/registry can build the backend before bootstrap; the socket is only
touched on the first real call).

Capabilities (design "MultiplexerCapabilities"): ``ids_stable_across_restart``
is False (a herdr *server* restart re-mints ids — Task 8 re-resolves via
session id), ``exposes_pane_tty`` is False (no tty in ``process-info`` on
macOS), ``native_agent_status`` and ``supports_event_stream`` are True,
``read_max_lines`` is 1000 (the ``pane read --source recent`` clamp).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

import structlog

from .base import (
    CaptureResult,
    ForegroundInfo,
    MultiplexerCapabilities,
    PaneDims,
    PaneInfo,
    WindowRef,
    WorkspaceRef,
)
from .topic_mapping import format_agent_topic_prefix

__all__ = [
    "HERDR_PROTOCOL_VERSION",
    "HerdrError",
    "HerdrManager",
    "HerdrProtocolError",
]

logger = structlog.get_logger()

# Pinned herdr socket protocol version (``herdr status`` → ``server.protocol``).
# herdr v0.7.0 speaks protocol 14. Bump deliberately after re-running the
# contract test against a newer herdr (design risk "herdr maturity").
HERDR_PROTOCOL_VERSION = 14

# Static capability declaration for the herdr backend (design Task 7).
_HERDR_CAPABILITIES = MultiplexerCapabilities(
    name="herdr",
    ids_stable_across_restart=False,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=1000,
    self_identify_env="HERDR_PANE_ID",
    supports_event_stream=True,
)

# Filter for self-hosted / internal workspaces and tabs (e.g. ``__main__``).
# Entries matching this pattern are skipped in ``list_windows`` so ccgram
# never auto-adopts itself. ``find_window`` deliberately bypasses this filter.
_INTERNAL_LABEL_RE = re.compile(r"^__.*__$")

# The send-keys path uses tmux key vocabulary ("Up"/"BSpace"/…); map the few
# that differ to herdr's kitty-style names. Unmapped tokens pass through.
_KEY_ALIASES: Mapping[str, str] = {
    "BSpace": "Backspace",
    "Space": "space",
}

# Runner contract: ``(returncode, stdout, stderr)``. Injectable for tests.
HerdrRunner = Callable[[Sequence[str]], "Awaitable[tuple[int, str, str]]"]

# Synthetic return codes from the default runner for non-exec failures.
_RC_TIMEOUT = 124
_RC_NO_BINARY = 127
_CALL_TIMEOUT_SECONDS = 8.0


class HerdrError(RuntimeError):
    """A herdr CLI/socket call failed (exit≠0, bad JSON, or an error payload)."""


class HerdrProtocolError(HerdrError):
    """The running herdr server speaks an unsupported protocol version."""


def _pane_index(pane_id: str) -> int:
    """Parse the integer pane number from a herdr ``wN:pM`` id (``M``)."""
    _, sep, num = pane_id.rpartition(":p")
    return int(num) if sep and num.isdigit() else 0


class HerdrManager:
    """Herdr backend satisfying the ``Multiplexer`` Protocol.

    Returns the neutral value types and exposes ``capabilities``. All herdr
    JSON parsing is private; methods return ``None``/``[]``/``False`` on failure
    exactly like the tmux backend, so callers gate on the result, never on a
    herdr-specific error type.
    """

    @property
    def capabilities(self) -> MultiplexerCapabilities:
        """Return the static capability declaration for the herdr backend."""
        return _HERDR_CAPABILITIES

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        binary: str = "herdr",
        runner: HerdrRunner | None = None,
    ) -> None:
        """Build the backend without touching the socket (I/O-free).

        Args:
            socket_path: herdr socket; defaults to ``$HERDR_SOCKET_PATH``.
            binary: the ``herdr`` executable name/path.
            runner: async ``(args) -> (rc, stdout, stderr)`` override for tests.
        """
        self._socket_path = socket_path or os.environ.get("HERDR_SOCKET_PATH", "")
        self._binary = binary
        self._run: HerdrRunner = runner or self._subprocess_run

    # ── CLI plumbing (private) ─────────────────────────────────────────

    async def _subprocess_run(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Default runner: exec ``herdr <args>`` with the socket env, time-boxed."""
        env = dict(os.environ)
        if self._socket_path:
            env["HERDR_SOCKET_PATH"] = self._socket_path
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            async with asyncio.timeout(_CALL_TIMEOUT_SECONDS):
                stdout, stderr = await proc.communicate()
        except TimeoutError:
            if proc:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
            return (_RC_TIMEOUT, "", "herdr call timed out")
        except OSError as exc:
            return (_RC_NO_BINARY, "", str(exc))
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _call_json(self, args: Sequence[str]) -> dict | None:
        """Run ``herdr <args>`` and return the JSON ``result`` dict, or None.

        None on: non-zero exit (socket down, bad id), non-JSON output, or an
        ``error`` payload. The failure is logged at debug — callers treat None
        as "window gone / call failed" (matches the tmux backend).
        """
        rc, out, err = await self._run(args)
        if rc != 0:
            logger.debug("herdr call failed", args=list(args), rc=rc, err=err.strip())
            return None
        try:
            payload = json.loads(out)
        except json.JSONDecodeError, ValueError:
            logger.debug("herdr returned non-JSON", args=list(args))
            return None
        if not isinstance(payload, dict):
            return None
        if "error" in payload:
            logger.debug("herdr error payload", args=list(args), error=payload["error"])
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    async def _call_ok(self, args: Sequence[str]) -> bool:
        """Run a mutating ``herdr`` command; True when it succeeded.

        Mutating commands vary in output: ``pane run`` / ``send-text`` /
        ``send-keys`` / ``report-metadata`` print nothing on success, while
        ``pane close`` / ``rename`` return a JSON envelope. A zero exit is
        success unless the JSON carries an ``error`` payload.
        """
        rc, out, err = await self._run(args)
        if rc != 0:
            logger.debug("herdr call failed", args=list(args), rc=rc, err=err.strip())
            return False
        text = out.strip()
        if not text:
            return True
        try:
            payload = json.loads(text)
        except json.JSONDecodeError, ValueError:
            return True  # non-JSON chatter on a zero exit → success
        return not (isinstance(payload, dict) and "error" in payload)

    async def _call_text(self, args: Sequence[str]) -> str | None:
        """Run ``herdr pane read`` (raw text on stdout); None on failure/empty."""
        rc, out, err = await self._run(args)
        if rc != 0:
            logger.debug("herdr read failed", args=list(args), rc=rc, err=err.strip())
            return None
        text = out.rstrip()
        return text or None

    async def _pane_get(self, pane_id: str) -> dict | None:
        """Return the private ``pane`` dict for a pane id, or None if gone."""
        result = await self._call_json(["pane", "get", pane_id])
        if not result:
            return None
        pane = result.get("pane")
        return pane if isinstance(pane, dict) else None

    async def _panes_for_tab(self, tab_id: str) -> list[dict]:
        """Return all pane dicts belonging to *tab_id* (one ``pane list`` call)."""
        pane_result = await self._call_json(["pane", "list"])
        if not pane_result:
            return []
        return [p for p in pane_result.get("panes", []) if p.get("tab_id") == tab_id]

    async def _active_pane(self, tab_id: str) -> str | None:
        """Resolve a tab id to its active pane id.

        Prefers the focused pane; falls back to the first pane in the tab.
        Returns ``None`` when the tab has no panes (gone or empty).
        """
        panes = await self._panes_for_tab(tab_id)
        if not panes:
            return None
        focused = next((p for p in panes if p.get("focused")), None)
        chosen = focused or panes[0]
        return chosen.get("pane_id") or None

    async def _tab_list(self) -> list[dict]:
        """Return the raw tab dicts from ``tab list`` (private, full objects)."""
        result = await self._call_json(["tab", "list"])
        if not result:
            return []
        return [t for t in result.get("tabs", []) if t.get("tab_id")]

    async def _tab_get(self, tab_id: str) -> dict | None:
        """Return the raw tab dict from ``tab get <tab_id>``; None when gone."""
        if not tab_id:
            return None
        result = await self._call_json(["tab", "get", tab_id])
        if not result:
            return None
        tab = result.get("tab")
        return tab if isinstance(tab, dict) else None

    async def _workspace_labels(self) -> dict[str, str]:
        """Map every ``workspace_id`` → its label (one ``workspace list`` call).

        Empty when herdr exposes no workspace addressing (older server) — the
        adaptive label then degrades to the agent name alone.
        """
        result = await self._call_json(["workspace", "list"])
        if not result:
            return {}
        return {
            w.get("workspace_id", ""): w.get("label", "")
            for w in result.get("workspaces", [])
            if w.get("workspace_id")
        }

    @staticmethod
    def _to_window_ref(
        tab_id: str,
        window_name: str,
        cwd: str,
        agent: str,
    ) -> WindowRef:
        """Build a neutral ``WindowRef`` from resolved tab fields.

        ``window_id`` is the ``tab_id`` (tab identity — design Task 1).
        ``window_name`` is the display label (full adaptive topic label
        ``"<workspace> ▸ <tab>"`` for both ``find_window`` and ``list_windows``).
        ``pane_current_command`` carries the representative agent label so
        provider detection and the status pipeline keep working.
        herdr has no tty and dimensions come from ``pane_dims`` on demand.
        """
        return WindowRef(
            window_id=tab_id,
            window_name=window_name or agent,
            cwd=cwd,
            pane_current_command=agent,
        )

    # ── Multiplexer Protocol surface ───────────────────────────────────

    async def ensure_session(self) -> None:
        """Verify the herdr server is reachable and speaks a pinned protocol.

        Raises:
            HerdrProtocolError: server protocol ≠ ``HERDR_PROTOCOL_VERSION``.
            HerdrError: socket unreachable / ``herdr status`` failed.
        """
        rc, out, err = await self._run(["status", "--json"])
        if rc != 0:
            raise HerdrError(f"herdr status failed: {err.strip() or f'exit {rc}'}")
        try:
            status = json.loads(out)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HerdrError("herdr status returned non-JSON") from exc
        if not isinstance(status, dict):
            raise HerdrError("herdr status returned non-object JSON")
        server = status.get("server") or {}
        if not server.get("running"):
            raise HerdrError("herdr server is not running")
        proto = server.get("protocol")
        if proto != HERDR_PROTOCOL_VERSION:
            raise HerdrProtocolError(
                f"herdr protocol {proto!r} unsupported "
                f"(ccgram pins {HERDR_PROTOCOL_VERSION})"
            )

    @staticmethod
    def _representative_pane(tab_panes: list[dict], tab_cwd: str) -> tuple[str, str]:
        """Return ``(agent, cwd)`` for the representative pane in *tab_panes*.

        Prefers the focused pane's agent; falls back to the first pane with a
        non-empty agent. ``tab_cwd`` is the fallback when no pane has a cwd.
        """
        focused = next((p for p in tab_panes if p.get("focused")), None)
        if focused:
            agent = focused.get("display_agent") or focused.get("agent", "")
            cwd = focused.get("cwd", "") or tab_cwd
            if agent:
                return agent, cwd
        for pane in tab_panes:
            candidate = pane.get("display_agent") or pane.get("agent", "")
            if candidate:
                return candidate, pane.get("cwd", "") or tab_cwd
        cwd = (focused or {}).get("cwd", "") or tab_cwd if focused else tab_cwd
        return "", cwd

    async def list_windows(self) -> list[WindowRef]:
        """List one ``WindowRef`` per herdr tab with its adaptive topic label.

        Identity: ``window_id = tab_id`` (tab identity — design Task 1). Builds
        from ``tab list`` + ``workspace list`` (labels) + ``pane list`` (per-tab
        representative agent and pane count). Representative agent = focused
        pane's ``agent``, else first non-empty.

        Tabs whose workspace or tab label matches ``__*__`` (e.g. ``__main__``)
        are skipped so ccgram never auto-adopts itself.

        This is the single source driving topic discovery and display-name
        re-sync: a workspace/tab rename re-labels the bound topic on the next
        poll without touching the binding key (agent session id, Task 2).
        """
        tabs = await self._tab_list()
        if not tabs:
            return []
        workspace_labels = await self._workspace_labels()

        # Build per-tab pane index from pane list (tab_id → list[pane]).
        pane_result = await self._call_json(["pane", "list"])
        panes_by_tab: dict[str, list[dict]] = {}
        if pane_result:
            for pane in pane_result.get("panes", []):
                tid = pane.get("tab_id", "")
                if tid:
                    panes_by_tab.setdefault(tid, []).append(pane)

        refs: list[WindowRef] = []
        for tab in tabs:
            tab_id = tab.get("tab_id", "")
            tab_label = tab.get("label", "")
            workspace_label = workspace_labels.get(tab.get("workspace_id", ""), "")

            # Skip __*__ workspace or tab labels.
            if _INTERNAL_LABEL_RE.match(workspace_label) or _INTERNAL_LABEL_RE.match(
                tab_label
            ):
                continue

            tab_panes = panes_by_tab.get(tab_id, [])
            rep_agent, rep_cwd = self._representative_pane(
                tab_panes, tab.get("cwd", "")
            )
            window_name = format_agent_topic_prefix(workspace_label, tab_label)
            refs.append(self._to_window_ref(tab_id, window_name, rep_cwd, rep_agent))
        return refs

    async def find_window(self, window_id: str) -> WindowRef | None:
        """Find a window by its tab id; None when gone.

        Uses ``tab get`` (tab identity — Task 1). Bypasses the ``__*__`` filter
        so an explicitly bound ``__*__`` tab still resolves for send/capture.
        cwd and representative agent come from the first available pane.
        Produces the same full ``"<workspace> ▸ <tab>"`` label as ``list_windows``
        so display-name consumers see a consistent topic title.
        """
        tab = await self._tab_get(window_id)
        if tab is None:
            return None
        tab_label = tab.get("label", "")

        # Resolve workspace label for the full adaptive topic label.
        workspace_labels = await self._workspace_labels()
        workspace_label = workspace_labels.get(tab.get("workspace_id", ""), "")
        window_name = format_agent_topic_prefix(workspace_label, tab_label)

        # Resolve cwd and agent from panes (tab get carries no pane detail).
        pane_result = await self._call_json(["pane", "list"])
        rep_agent = ""
        rep_cwd = tab.get("cwd", "")
        if pane_result:
            tab_panes = [
                p for p in pane_result.get("panes", []) if p.get("tab_id") == window_id
            ]
            focused = next((p for p in tab_panes if p.get("focused")), None)
            rep_pane = focused or (tab_panes[0] if tab_panes else None)
            if rep_pane:
                rep_agent = rep_pane.get("display_agent") or rep_pane.get("agent", "")
                rep_cwd = rep_pane.get("cwd", "") or rep_cwd

        return self._to_window_ref(window_id, window_name, rep_cwd, rep_agent)

    # ── Raw pane-id ops (private) ──────────────────────────────────────
    # These accept a resolved *pane* id — not a tab id. Tab-keyed public
    # methods resolve tab→active-pane via ``_active_pane`` before calling here.

    async def _read_visible_pane(
        self, pane_id: str, *, ansi: bool = False
    ) -> str | None:
        """Read visible pane text for a resolved pane id; None on failure."""
        fmt = "ansi" if ansi else "text"
        return await self._call_text(
            ["pane", "read", pane_id, "--source", "visible", "--format", fmt]
        )

    async def _read_recent_pane(self, pane_id: str, *, lines: int) -> str | None:
        """Read recent scrollback for a resolved pane id; None on failure."""
        return await self._call_text(
            [
                "pane",
                "read",
                pane_id,
                "--source",
                "recent",
                "--lines",
                str(lines),
                "--format",
                "text",
            ]
        )

    async def _dims_for_pane(self, pane_id: str) -> PaneDims | None:
        """Return dimensions for a resolved pane id from ``pane layout``."""
        result = await self._call_json(["pane", "layout", "--pane", pane_id])
        if not result:
            return None
        layout = result.get("layout") or {}
        for pane in layout.get("panes", []):
            if pane.get("pane_id") == pane_id:
                rect = pane.get("rect") or {}
                w, h = rect.get("width"), rect.get("height")
                if isinstance(w, int) and isinstance(h, int):
                    return PaneDims(width=w, height=h)
        area = layout.get("area") or {}
        w, h = area.get("width"), area.get("height")
        if isinstance(w, int) and isinstance(h, int):
            return PaneDims(width=w, height=h)
        return None

    async def _foreground_for_pane(self, pane_id: str) -> ForegroundInfo | None:
        """Return foreground process info for a resolved pane id."""
        result = await self._call_json(["pane", "process-info", "--pane", pane_id])
        if not result:
            return None
        info = result.get("process_info") or {}
        procs = info.get("foreground_processes") or []
        if not procs:
            return None
        pgid = info.get("foreground_process_group_id") or 0
        leader = next((p for p in procs if p.get("pid") == pgid), procs[0])
        return ForegroundInfo(
            pid=int(leader.get("pid", 0)),
            pgid=int(pgid or leader.get("pid", 0)),
            argv=list(leader.get("argv") or []),
            cwd=leader.get("cwd", "") or "",
            tty="",
        )

    # ── Tab-keyed public ops (resolve tab→active-pane first) ───────────

    async def capture(
        self, window_id: str, *, ansi: bool = False
    ) -> CaptureResult | None:
        """Capture visible pane text.

        *window_id* is a tab id. Resolves the tab to its active pane first,
        then reads visible text via ``pane read --source visible``.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return None
        text = await self._read_visible_pane(pane_id, ansi=ansi)
        if text is None:
            return None
        return CaptureResult(text=text)

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Capture recent scrollback, clamped to ``read_max_lines`` (1000).

        *window_id* is a tab id. Resolves to the active pane first.
        ``truncated`` is True when the caller asked for more lines than herdr
        will return.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return None
        max_lines = self.capabilities.read_max_lines
        effective = lines
        truncated = False
        if max_lines is not None and lines > max_lines:
            effective = max_lines
            truncated = True
        text = await self._read_recent_pane(pane_id, lines=effective)
        if text is None:
            return None
        return CaptureResult(text=text, truncated=truncated)

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Return the active pane's columns/rows from ``pane layout``.

        *window_id* is a tab id. Resolves to the active pane first.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return None
        return await self._dims_for_pane(pane_id)

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,  # noqa: ARG002  # pyright: ignore[reportUnusedVariable]
    ) -> bool:
        """Send text/keys to the active pane in a tab.

        *window_id* is a tab id. Resolves to the active pane first.
        ``literal``+``enter`` → ``pane run`` (atomic text+Enter); ``literal``
        without ``enter`` → ``pane send-text``; ``literal=False`` treats *text*
        as space-separated key names → ``pane send-keys``. herdr needs no vim
        workaround, so ``raw`` is accepted for parity and ignored.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return False
        return await self._send_to(pane_id, text, enter=enter, literal=literal)

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,  # noqa: ARG002 — protocol signature
    ) -> bool:
        """Send to a specific pane id directly (no tab resolution).

        Unlike ``send``, *pane_id* here is a real herdr pane id (e.g.
        ``"w2:p1"``), not a tab id — callers that target a specific pane in a
        split tab pass the pane id directly.
        """
        return await self._send_to(pane_id, text, enter=enter, literal=literal)

    async def _send_to(
        self, pane_id: str, text: str, *, enter: bool, literal: bool
    ) -> bool:
        if not literal:
            keys = [_KEY_ALIASES.get(tok, tok) for tok in text.split() if tok]
            if enter:
                keys.append("Enter")
            if not keys:
                return False
            return await self._call_ok(["pane", "send-keys", pane_id, *keys])
        if enter:
            return await self._call_ok(["pane", "run", pane_id, text])
        return await self._call_ok(["pane", "send-text", pane_id, text])

    async def kill_window(self, window_id: str) -> bool:
        """Close a herdr tab (``tab close``).

        ``window_id`` is a tab id (tab identity — Task 1).
        """
        ok = await self._call_ok(["tab", "close", window_id])
        if ok:
            logger.info("Closed herdr tab %s", window_id)
        return ok

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a herdr tab (``tab rename``).

        ``window_id`` is a tab id (tab identity — Task 1).
        """
        return await self._call_ok(["tab", "rename", window_id, new_name])

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """Return ALL panes in a herdr tab (team awareness).

        *window_id* is a tab id. Fetches all panes in the tab from ``pane list``
        and resolves per-pane dimensions from a single ``pane layout`` call.
        Returns ``[]`` when the tab has no panes.
        """
        panes = await self._panes_for_tab(window_id)
        if not panes:
            return []
        # Fetch layout once for the active pane; extract per-pane rects.
        active_pane_id = next(
            (p.get("pane_id", "") for p in panes if p.get("focused")),
            panes[0].get("pane_id", ""),
        )
        layout_result = await self._call_json(
            ["pane", "layout", "--pane", active_pane_id]
        )
        layout_rects: dict[str, PaneDims] = {}
        if layout_result:
            layout = layout_result.get("layout") or {}
            area = layout.get("area") or {}
            area_w, area_h = area.get("width", 0), area.get("height", 0)
            for lp in layout.get("panes", []):
                pid = lp.get("pane_id", "")
                if pid:
                    rect = lp.get("rect") or {}
                    lw = rect.get("width")
                    lh = rect.get("height")
                    if isinstance(lw, int) and isinstance(lh, int):
                        layout_rects[pid] = PaneDims(width=lw, height=lh)
                    elif isinstance(area_w, int) and isinstance(area_h, int):
                        layout_rects[pid] = PaneDims(width=area_w, height=area_h)
        result: list[PaneInfo] = []
        for pane in panes:
            pid = pane.get("pane_id", "")
            dims = layout_rects.get(pid)
            result.append(
                PaneInfo(
                    pane_id=pid,
                    index=_pane_index(pid),
                    active=bool(pane.get("focused", False)),
                    command=pane.get("agent", ""),
                    path=pane.get("cwd", ""),
                    width=dims.width if dims else 0,
                    height=dims.height if dims else 0,
                )
            )
        return result

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """List all herdr workspaces as neutral ``WorkspaceRef`` objects.

        Returns ``[]`` when the workspace command is unavailable (older herdr
        server) — callers must handle the empty case gracefully (fall through
        to cwd-resolve).
        """
        result = await self._call_json(["workspace", "list"])
        if not result:
            return []
        return [
            WorkspaceRef(
                workspace_id=ws.get("workspace_id", ""),
                label=ws.get("label", ""),
                cwd=ws.get("cwd", ""),
            )
            for ws in result.get("workspaces", [])
            if ws.get("workspace_id")
        ]

    async def _resolve_workspace_id(self, cwd: str) -> str:
        """Return the workspace rooted at *cwd*, creating one if none matches.

        Reuses the herdr workspace whose cwd matches the target directory so a
        new agent lands in the repo's existing workspace and inherits its label
        as the topic prefix (design "cwd → workspace"). Returns "" when herdr
        exposes no workspace addressing (older server / command unavailable) —
        ``create_window`` then falls back to a plain ``tab create`` in the
        active workspace (Task 7 behavior).
        """
        result = await self._call_json(["workspace", "list"])
        if result:
            for ws in result.get("workspaces", []):
                if self._same_path(ws.get("cwd", ""), cwd):
                    wid = ws.get("workspace_id", "")
                    if wid:
                        return wid
        created = await self._call_json(["workspace", "create", "--cwd", cwd])
        if not created:
            return ""
        return (created.get("workspace") or {}).get("workspace_id", "") or ""

    @staticmethod
    def _same_path(a: str, b: str) -> bool:
        """True when two paths point at the same directory (symlinks resolved)."""
        if not a or not b:
            return False
        try:
            return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
        except OSError:
            return a == b

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
        """Create a herdr tab at *work_dir* and optionally launch an agent.

        Resolves *work_dir* to its herdr workspace (reusing the matching one,
        creating it only if absent — design "cwd → workspace"), creates a
        ``tab`` inside it, then ``pane run``s the launch command in the root
        pane.

        When *workspace_id* is provided (from the UI workspace picker), the
        cwd-resolve step is skipped and the tab is created inside that workspace
        directly.

        Returns ``(success, message, window_name, window_id)`` where
        ``window_id`` is the new **tab id** (tab identity — Task 1). The agent
        launch still targets the root pane id (pane ops are resolved via
        tab→active-pane in Task 4).
        """
        path = Path(work_dir).expanduser()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", "", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", "", ""

        cwd = str(path)
        if workspace_id is None:
            workspace_id = await self._resolve_workspace_id(cwd)
        args = ["tab", "create", "--cwd", cwd, "--no-focus"]
        if workspace_id:
            args += ["--workspace", workspace_id]
        if window_name:
            args += ["--label", window_name]
        result = await self._call_json(args)
        if not result:
            return False, f"Failed to create herdr tab at {path}", "", ""

        tab = result.get("tab") or {}
        tab_id = tab.get("tab_id", "")
        label = tab.get("label", window_name or "")
        if not tab_id:
            return False, "herdr tab created without a tab id", "", ""

        if start_agent and launch_command:
            root_pane = result.get("root_pane") or {}
            pane_id = root_pane.get("pane_id", "")
            if pane_id:
                cmd = (
                    f"{launch_command} {agent_args}".strip()
                    if agent_args
                    else launch_command
                )
                await self._call_ok(["pane", "run", pane_id, cmd])

        logger.info("Created herdr tab %r (id=%s) at %s", label, tab_id, path)
        return True, f"Created herdr tab '{label}' at {path}", label, tab_id

    async def set_title(self, window_id: str, provider_name: str) -> None:
        """Stamp the active pane title for instant provider re-detection.

        *window_id* is a tab id. Resolves to the active pane first.
        Uses ``pane report-metadata --title ccgram:<provider>`` (herdr's
        title channel); best-effort, failures are swallowed like tmux.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return
        await self._call_ok(
            [
                "pane",
                "report-metadata",
                pane_id,
                "--source",
                "ccgram",
                "--title",
                f"ccgram:{provider_name}",
            ]
        )

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """Foreground process info for the active pane in a tab.

        *window_id* is a tab id. Resolves to the active pane first.
        No ``ps -t`` and no tty (``exposes_pane_tty`` is False — macOS herdr
        reports no tty). Picks the process-group leader, else the first
        foreground process.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return None
        return await self._foreground_for_pane(pane_id)

    # ── Transitional surface (legacy aliases) ──────────────────────────
    # Mirror the historical ``tmux_manager`` names callers still use, so the
    # herdr backend satisfies the same contract (F2) without rewriting callers.

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """Legacy alias of ``find_window``."""
        return await self.find_window(window_id)

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Visible pane text as a plain string (legacy alias of ``capture``)."""
        result = await self.capture(window_id, ansi=with_ansi)
        return result.text if result else None

    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,  # noqa: ARG002 — protocol signature
    ) -> str | None:
        """Capture a specific pane's visible text by pane id (no tab resolution).

        *pane_id* is a real herdr pane id (e.g. ``"w2:p1"``). Reads directly
        without resolving through a tab so callers that target a specific pane
        in a split tab get the right pane, not the active one.
        """
        return await self._read_visible_pane(pane_id, ansi=with_ansi)

    async def capture_pane_scrollback(
        self, window_id: str, history: int = 200
    ) -> str | None:
        """Scrollback text as a plain string (legacy alias)."""
        result = await self.capture_scrollback(window_id, lines=history)
        return result.text if result else None

    async def send_keys(
        self,
        window_id: str,
        text: str,
        enter: bool = True,
        literal: bool = True,
        *,
        raw: bool = False,
    ) -> bool:
        """Legacy alias of ``send``."""
        return await self.send(window_id, text, enter=enter, literal=literal, raw=raw)

    async def send_keys_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Legacy alias of ``send_to_pane``."""
        return await self.send_to_pane(
            pane_id, text, enter=enter, literal=literal, window_id=window_id
        )

    async def get_pane_title(self, window_id: str) -> str:
        """Return the active pane's reported title.

        *window_id* is a tab id. Resolves to the active pane first, then reads
        ``pane get`` → ``title``.
        """
        pane_id = await self._active_pane(window_id)
        if pane_id is None:
            return ""
        pane = await self._pane_get(pane_id)
        if pane is None:
            return ""
        return pane.get("title", "") or ""

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """Legacy alias of ``set_title``."""
        await self.set_title(window_id, provider_name)
