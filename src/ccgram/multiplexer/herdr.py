"""Herdr backend for the Multiplexer contract, via the herdr CLI/socket.

Anti-corruption layer over `herdr <https://github.com/ogulcancelik/herdr>`_'s
Unix-socket JSON-RPC CLI. Every herdr JSON shape (``pane_info`` / ``pane_list``
/ ``pane_process_info`` / ``pane_layout`` / ``tab_created`` …) and every
``wN:pN`` id string stays **private** to this module; callers see only the
neutral value types from ``multiplexer.base`` (design "Module map": herdr.py is
adapter, anti-corruption).

Identity mapping is trivial: herdr's opaque ``pane_id`` (``"w2:p1"``) *is* the
``window_id`` string (design decision 1, thin identity). One herdr agent pane ≈
one ccgram window.

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

    async def _pane_get(self, window_id: str) -> dict | None:
        """Return the private ``pane`` dict for a window id, or None if gone."""
        result = await self._call_json(["pane", "get", window_id])
        if not result:
            return None
        pane = result.get("pane")
        return pane if isinstance(pane, dict) else None

    async def _tab_labels(self) -> dict[str, str]:
        """Map every ``tab_id`` → its label (one ``tab list`` call)."""
        result = await self._call_json(["tab", "list"])
        if not result:
            return {}
        return {
            t.get("tab_id", ""): t.get("label", "")
            for t in result.get("tabs", [])
            if t.get("tab_id")
        }

    async def _tab_label(self, tab_id: str) -> str:
        """Return one tab's label, or '' when missing."""
        if not tab_id:
            return ""
        result = await self._call_json(["tab", "get", tab_id])
        if not result:
            return ""
        return (result.get("tab") or {}).get("label", "") or ""

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
    def _adaptive_label(
        pane: Mapping,
        tab_labels: Mapping[str, str],
        workspace_labels: Mapping[str, str],
        pane_counts: Mapping[str, int],
    ) -> str:
        """Build a pane's adaptive topic label from the herdr label maps.

        ``"<workspace> ▸ <agent>"`` (+ ``"/<tab>"`` when the pane's tab is
        split). Agent label prefers ``display_agent`` over the bare ``agent``.
        """
        tab_id = pane.get("tab_id", "")
        workspace = workspace_labels.get(pane.get("workspace_id", ""), "")
        agent = pane.get("display_agent") or pane.get("agent", "")
        split = pane_counts.get(tab_id, 1) > 1
        return format_agent_topic_prefix(
            workspace, agent, tab_labels.get(tab_id, ""), split=split
        )

    @staticmethod
    def _to_window_ref(pane: Mapping, window_name: str) -> WindowRef:
        """Project a private herdr pane dict onto the neutral ``WindowRef``.

        ``window_name`` is the display label the caller resolved (a cheap tab
        label for ``find_window``, the full adaptive topic label for
        ``list_windows``); ``pane_current_command`` carries the agent label so
        provider detection and status code keep working. herdr has no tty and
        dimensions come from ``pane_dims`` on demand, so ``pane_tty``/width/
        height stay defaults.
        """
        return WindowRef(
            window_id=pane.get("pane_id", ""),
            window_name=window_name or pane.get("agent", ""),
            cwd=pane.get("cwd", ""),
            pane_current_command=pane.get("agent", ""),
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

    async def list_windows(self) -> list[WindowRef]:
        """List every agent pane as a window with its adaptive topic label.

        ``window_name`` is the derived topic label ``"<workspace> ▸ <agent>"``
        (+ ``"/<tab>"`` when the pane's tab is split), sourced from one
        ``pane list`` + ``tab list`` + ``workspace list`` plus the per-tab pane
        counts. This is the single source that drives both topic discovery (the
        session monitor) and display-name re-sync, so a workspace/tab rename
        re-labels the bound topic on the next poll — the binding key (the
        durable agent session id, Task 8) is never touched.
        """
        result = await self._call_json(["pane", "list"])
        if not result:
            return []
        panes = [p for p in result.get("panes", []) if p.get("pane_id")]
        tab_labels = await self._tab_labels()
        workspace_labels = await self._workspace_labels()
        pane_counts: dict[str, int] = {}
        for pane in panes:
            tab_id = pane.get("tab_id", "")
            pane_counts[tab_id] = pane_counts.get(tab_id, 0) + 1
        return [
            self._to_window_ref(
                pane,
                self._adaptive_label(pane, tab_labels, workspace_labels, pane_counts),
            )
            for pane in panes
        ]

    async def find_window(self, window_id: str) -> WindowRef | None:
        """Find a window by its opaque pane id; None when gone."""
        pane = await self._pane_get(window_id)
        if pane is None:
            return None
        return self._to_window_ref(pane, await self._tab_label(pane.get("tab_id", "")))

    async def capture(
        self, window_id: str, *, ansi: bool = False
    ) -> CaptureResult | None:
        """Capture visible pane text (``pane read --source visible``)."""
        fmt = "ansi" if ansi else "text"
        text = await self._call_text(
            ["pane", "read", window_id, "--source", "visible", "--format", fmt]
        )
        if text is None:
            return None
        return CaptureResult(text=text)

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Capture recent scrollback, clamped to ``read_max_lines`` (1000).

        ``truncated`` is True when the caller asked for more lines than herdr
        will return.
        """
        max_lines = self.capabilities.read_max_lines
        effective = lines
        truncated = False
        if max_lines is not None and lines > max_lines:
            effective = max_lines
            truncated = True
        text = await self._call_text(
            [
                "pane",
                "read",
                window_id,
                "--source",
                "recent",
                "--lines",
                str(effective),
                "--format",
                "text",
            ]
        )
        if text is None:
            return None
        return CaptureResult(text=text, truncated=truncated)

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Return the pane's columns/rows from ``pane layout``; None if gone."""
        result = await self._call_json(["pane", "layout", "--pane", window_id])
        if not result:
            return None
        layout = result.get("layout") or {}
        for pane in layout.get("panes", []):
            if pane.get("pane_id") == window_id:
                rect = pane.get("rect") or {}
                w, h = rect.get("width"), rect.get("height")
                if isinstance(w, int) and isinstance(h, int):
                    return PaneDims(width=w, height=h)
        area = layout.get("area") or {}
        w, h = area.get("width"), area.get("height")
        if isinstance(w, int) and isinstance(h, int):
            return PaneDims(width=w, height=h)
        return None

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,  # noqa: ARG002 — protocol signature; herdr needs no raw workaround
    ) -> bool:
        """Send text/keys to a pane.

        ``literal``+``enter`` → ``pane run`` (atomic text+Enter); ``literal``
        without ``enter`` → ``pane send-text``; ``literal=False`` treats *text*
        as space-separated key names → ``pane send-keys``. herdr needs no vim
        workaround, so ``raw`` is accepted for parity and ignored.
        """
        return await self._send_to(window_id, text, enter=enter, literal=literal)

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,  # noqa: ARG002 — protocol signature; herdr panes are single-pane windows
    ) -> bool:
        """Send to a specific pane id.

        herdr panes are globally addressable single-pane windows, so the
        ``window_id`` cross-window guard is a no-op here (each pane *is* its
        own window).
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
        """Close a herdr pane (``pane close``)."""
        ok = await self._call_ok(["pane", "close", window_id])
        if ok:
            logger.info("Closed herdr pane %s", window_id)
        return ok

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Set a pane's label (``pane rename``)."""
        return await self._call_ok(["pane", "rename", window_id, new_name])

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """Return the window's pane(s).

        A herdr window is a single agent pane (tab splits surface as separate
        windows/topics — design "topic = pane = agent"), so this returns a
        one-element list, or ``[]`` when the pane is gone.
        """
        pane = await self._pane_get(window_id)
        if pane is None:
            return []
        dims = await self.pane_dims(window_id)
        return [
            PaneInfo(
                pane_id=pane.get("pane_id", ""),
                index=_pane_index(pane.get("pane_id", "")),
                active=bool(pane.get("focused", False)),
                command=pane.get("agent", ""),
                path=pane.get("cwd", ""),
                width=dims.width if dims else 0,
                height=dims.height if dims else 0,
            )
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
    ) -> tuple[bool, str, str, str]:
        """Create a herdr tab at *work_dir* and optionally launch an agent.

        Resolves *work_dir* to its herdr workspace (reusing the matching one,
        creating it only if absent — design "cwd → workspace"), creates a
        ``tab`` inside it (its root pane becomes the window), then ``pane run``s
        the launch command.

        Returns ``(success, message, window_name, window_id)`` where
        ``window_id`` is the new pane id.
        """
        path = Path(work_dir).expanduser()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", "", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", "", ""

        cwd = str(path)
        workspace_id = await self._resolve_workspace_id(cwd)
        args = ["tab", "create", "--cwd", cwd, "--no-focus"]
        if workspace_id:
            args += ["--workspace", workspace_id]
        if window_name:
            args += ["--label", window_name]
        result = await self._call_json(args)
        if not result:
            return False, f"Failed to create herdr tab at {path}", "", ""

        root_pane = result.get("root_pane") or {}
        pane_id = root_pane.get("pane_id", "")
        label = (result.get("tab") or {}).get("label", window_name or "")
        if not pane_id:
            return False, "herdr tab created without a pane", "", ""

        if start_agent and launch_command:
            cmd = (
                f"{launch_command} {agent_args}".strip()
                if agent_args
                else launch_command
            )
            await self._call_ok(["pane", "run", pane_id, cmd])

        logger.info("Created herdr tab %r (pane=%s) at %s", label, pane_id, path)
        return True, f"Created herdr tab '{label}' at {path}", label, pane_id

    async def set_title(self, window_id: str, provider_name: str) -> None:
        """Stamp the pane title for instant provider re-detection.

        Uses ``pane report-metadata --title ccgram:<provider>`` (herdr's
        title channel); best-effort, failures are swallowed like tmux.
        """
        await self._call_ok(
            [
                "pane",
                "report-metadata",
                window_id,
                "--source",
                "ccgram",
                "--title",
                f"ccgram:{provider_name}",
            ]
        )

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """Foreground process info from ``pane process-info``.

        No ``ps -t`` and no tty (``exposes_pane_tty`` is False — macOS herdr
        reports no tty). Picks the process-group leader, else the first
        foreground process.
        """
        result = await self._call_json(["pane", "process-info", "--pane", window_id])
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
        window_id: str | None = None,  # noqa: ARG002 — protocol signature; herdr panes are single-pane windows
    ) -> str | None:
        """Capture a specific pane's visible text by id (herdr: pane == window)."""
        result = await self.capture(pane_id, ansi=with_ansi)
        return result.text if result else None

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
        """Return the pane's reported title (herdr ``pane get`` → ``title``)."""
        pane = await self._pane_get(window_id)
        if pane is None:
            return ""
        return pane.get("title", "") or ""

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """Legacy alias of ``set_title``."""
        await self.set_title(window_id, provider_name)
