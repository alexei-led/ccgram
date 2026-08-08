"""Herdr backend for the Multiplexer contract, via the herdr CLI/socket.

Anti-corruption layer over `herdr <https://github.com/ogulcancelik/herdr>`_'s
Unix-socket JSON-RPC CLI. Every herdr JSON shape (``pane_info`` / ``pane_list``
/ ``pane_process_info`` / ``pane_layout`` / ``tab_created`` …) and every
``wN:pN``/``wN:tN`` id string stays **private** to this module; callers see
only the neutral value types from ``multiplexer.base`` (design "Module map":
herdr.py is adapter, anti-corruption).

Identity mapping: Herdr ``agent.list`` is the sole identity source.  One
complete agent-session composite becomes an opaque durable target; tab, pane,
workspace, and terminal identifiers are short-lived locators used only after
a fresh guard authorizes one action.

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
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path

import structlog

from ..herdr_targets import is_herdr_session_target
from .base import (
    AgentStatus,
    CaptureResult,
    ForegroundInfo,
    MultiplexerCapabilities,
    MuxEvent,
    PaneDims,
    PaneInfo,
    TopicTargetResult,
    WindowRef,
    WorkspaceRef,
)
from .herdr_events import (
    is_subscribed_sentinel,
    open_socket_stream,
    translate_event,
)
from .topic_mapping import format_agent_topic_prefix

__all__ = [
    "HERDR_PROTOCOL_VERSION",
    "HERDR_SUPPORTED_PROTOCOLS",
    "HerdrAgentListError",
    "HerdrAmbiguousTargetError",
    "HerdrError",
    "HerdrLiveRecord",
    "HerdrMalformedRecordError",
    "HerdrManager",
    "HerdrProtocolError",
    "HerdrSessionComposite",
    "HerdrUnresolvedTargetError",
    "canonical_session_bytes",
    "herdr_session_target_id",
]

logger = structlog.get_logger()

# Supported herdr socket protocols (``herdr status`` → ``server.protocol``).
# 14–17 and 19 are supported. Other versions are attempted with a warning so
# ccgram remains usable across herdr upgrades and downgrades.
HERDR_SUPPORTED_PROTOCOLS = frozenset({14, 15, 16, 17, 19})
HERDR_PROTOCOL_VERSION = max(HERDR_SUPPORTED_PROTOCOLS)

# Static capability declaration for the herdr backend (design Task 7).
_HERDR_CAPABILITIES = MultiplexerCapabilities(
    name="herdr",
    ids_stable_across_restart=False,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=1000,
    self_identify_env="HERDR_PANE_ID",
    supports_event_stream=True,
    native_worktrees=True,
)

# Filter for self-hosted / internal workspaces and tabs (e.g. ``__main__``).
# Entries matching this pattern are skipped in ``list_windows`` so ccgram
# never auto-adopts itself. ``find_window_by_id`` deliberately bypasses it.
_INTERNAL_LABEL_RE = re.compile(r"^__.*__$")

# The send-keys path uses tmux key vocabulary ("Up"/"BSpace"/…); map the few
# that differ to herdr's kitty-style names. Unmapped tokens pass through.
_KEY_ALIASES: Mapping[str, str] = {
    "BSpace": "Backspace",
    "Space": "space",
}

# Runner contract: ``(returncode, stdout, stderr)``. Injectable for tests.
HerdrRunner = Callable[[Sequence[str]], "Awaitable[tuple[int, str, str]]"]

# Stream-opener contract: ``(subscriptions) -> async iterator of event dicts``.
# Injectable for tests so ``watch_events`` can be driven with canned event lines
# (no socket). The default opens the live unix socket via ``open_socket_stream``.
HerdrStreamOpener = Callable[
    [Sequence[Mapping[str, object]]], "AsyncGenerator[dict, None]"
]

# Synthetic return codes from the default runner for non-exec failures.
_RC_TIMEOUT = 124
_RC_NO_BINARY = 127
_CALL_TIMEOUT_SECONDS = 8.0

# New Pi sessions have been observed to publish their agent_session in ~2.7s.
# Keep creation discovery bounded, while allowing slow hook/integration startup.
_CREATED_SESSION_DISCOVERY_TIMEOUT_SECONDS = 5.0
_CREATED_SESSION_POLL_INTERVAL_SECONDS = 0.1

# Event-stream reconnect backoff (seconds): exponential, capped.
_STREAM_BACKOFF_BASE = 1.0
_STREAM_BACKOFF_MAX = 30.0
# A live stream has no locator-change notification. Re-prime periodically so a
# target that moved to another pane receives a fresh per-pane subscription.
_STREAM_REPRIME_INTERVAL = 5.0


def _workspace_cwd_from_panes(
    workspace: Mapping[str, object], panes: Sequence[Mapping[str, object]]
) -> str | None:
    """Return the active tab's sole pane CWD from a protocol-19 snapshot."""
    workspace_id = workspace.get("workspace_id")
    if not isinstance(workspace_id, str):
        return None
    active_tab_id = workspace.get("active_tab_id")
    candidates = [
        pane
        for pane in panes
        if pane.get("workspace_id") == workspace_id
        and (
            pane.get("tab_id") == active_tab_id
            if isinstance(active_tab_id, str)
            else bool(pane.get("focused"))
        )
    ]
    if len(candidates) != 1:
        return None
    candidate_cwd = candidates[0].get("foreground_cwd")
    if not isinstance(candidate_cwd, str):
        candidate_cwd = candidates[0].get("cwd")
    return candidate_cwd if isinstance(candidate_cwd, str) else None


class HerdrError(RuntimeError):
    """A herdr CLI/socket call failed (exit≠0, bad JSON, or an error payload)."""


class HerdrProtocolError(HerdrError):
    """Reserved for callers that require a strict herdr protocol policy."""


class HerdrAgentListError(HerdrError):
    """The fresh ``agent.list`` snapshot could not be read."""


class HerdrMalformedRecordError(HerdrError):
    """An ``agent.list`` record is not safe to use as a session target."""


class HerdrUnresolvedTargetError(HerdrError):
    """No current session record matches the requested target ID."""


class HerdrAmbiguousTargetError(HerdrError):
    """More than one current session record matches the requested target ID."""


@dataclass(frozen=True)
class HerdrSessionComposite:
    """The complete Herdr identity independent of its current locator."""

    source: str
    agent: str
    kind: str
    value: str


@dataclass(frozen=True)
class HerdrLiveRecord:
    """One sessionful agent and its short-lived current Herdr locator."""

    target_id: str
    composite: HerdrSessionComposite
    terminal_id: str
    pane_id: str
    tab_id: str
    workspace_id: str


def _session_field(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _session_composite(record: Mapping[str, object]) -> HerdrSessionComposite | None:
    """Parse one complete ``agent_session`` value, ignoring sessionless agents."""
    session = record.get("agent_session")
    if session is None:
        return None
    if not isinstance(session, Mapping):
        raise HerdrMalformedRecordError("agent.list contains a malformed agent_session")
    values = {
        key: _session_field(session.get(key))
        for key in ("source", "agent", "kind", "value")
    }
    if any(value is None for value in values.values()):
        raise HerdrMalformedRecordError(
            "agent.list contains an incomplete agent_session"
        )
    return HerdrSessionComposite(
        source=values["source"] or "",
        agent=values["agent"] or "",
        kind=values["kind"] or "",
        value=values["value"] or "",
    )


def canonical_session_bytes(composite: HerdrSessionComposite) -> bytes:
    """Return canonical UTF-8 bytes for a complete session composite."""
    values = {
        "source": composite.source,
        "agent": composite.agent,
        "kind": composite.kind,
        "value": composite.value,
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise HerdrMalformedRecordError("session composite is incomplete")
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return payload.encode("utf-8")


def herdr_session_target_id(composite: HerdrSessionComposite) -> str:
    """Return the opaque versioned ID for a complete session composite."""
    prefix = b"ccgram-herdr-session-v1\0"
    digest = hashlib.sha256(prefix + canonical_session_bytes(composite)).hexdigest()
    return f"herdr-session-v1-{digest}"


def _parse_live_record(record: Mapping[str, object]) -> HerdrLiveRecord | None:
    composite = _session_composite(record)
    if composite is None:
        return None
    locators = {
        key: _session_field(record.get(key))
        for key in ("terminal_id", "pane_id", "tab_id", "workspace_id")
    }
    if any(value is None for value in locators.values()):
        raise HerdrMalformedRecordError(
            "agent.list contains an incomplete live locator"
        )
    return HerdrLiveRecord(
        target_id=herdr_session_target_id(composite),
        composite=composite,
        terminal_id=locators["terminal_id"] or "",
        pane_id=locators["pane_id"] or "",
        tab_id=locators["tab_id"] or "",
        workspace_id=locators["workspace_id"] or "",
    )


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
        stream_opener: HerdrStreamOpener | None = None,
    ) -> None:
        """Build the backend without touching the socket (I/O-free).

        Args:
            socket_path: herdr socket; defaults to ``$HERDR_SOCKET_PATH``.
            binary: the ``herdr`` executable name/path.
            runner: async ``(args) -> (rc, stdout, stderr)`` override for tests.
            stream_opener: event-stream opener override for tests; defaults to
                the live unix-socket reader (``open_socket_stream``).
        """
        self._socket_path = socket_path or os.environ.get("HERDR_SOCKET_PATH", "")
        # Resolve to an absolute path: CPython only takes the fork-free
        # ``posix_spawn`` fast path when the executable has a dirname (see
        # subprocess.Popen._execute_child). Bare names force fork_exec, which
        # triggers macOS ``MallocStackLogging`` spam from long-lived parents.
        self._binary = shutil.which(binary) or binary
        self._run: HerdrRunner = runner or self._subprocess_run
        self._open_stream: HerdrStreamOpener = stream_opener or self._default_stream

    def _default_stream(
        self, subscriptions: Sequence[Mapping[str, object]]
    ) -> AsyncGenerator[dict, None]:
        """Open the live herdr socket and subscribe (default stream opener)."""
        return open_socket_stream(self._socket_path, subscriptions)

    # ── CLI plumbing (private) ─────────────────────────────────────────

    async def _subprocess_run(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Default runner: exec ``herdr <args>`` with the socket env, time-boxed."""
        env = dict(os.environ)
        if self._socket_path:
            env["HERDR_SOCKET_PATH"] = self._socket_path
        try:
            # Force CPython's fork-free ``posix_spawn`` path: it requires an
            # absolute executable (resolved in ``__init__``) and, on macOS
            # builds without ``posix_spawn_file_actions_addclosefrom_np``,
            # ``close_fds=False``. Forking from this long-lived async process
            # makes every child print macOS ``MallocStackLogging`` warnings.
            # fd inheritance is acceptable: herdr is a trusted, short-lived
            # CLI that only talks to its socket.
            completed = await asyncio.to_thread(
                subprocess.run,
                [self._binary, *args],
                capture_output=True,
                text=True,
                env=env,
                timeout=_CALL_TIMEOUT_SECONDS,
                check=False,
                close_fds=False,
            )
        except subprocess.TimeoutExpired:
            return (_RC_TIMEOUT, "", "herdr call timed out")
        except OSError as exc:
            return (_RC_NO_BINARY, "", str(exc))
        return (completed.returncode, completed.stdout, completed.stderr)

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

    # ── Multiplexer Protocol surface ───────────────────────────────────

    async def ensure_session(self) -> None:
        """Verify the herdr server is reachable; warn for unverified protocols.

        ``HERDR_SUPPORTED_PROTOCOLS`` are accepted without a warning. Other
        protocol versions are best-effort: ccgram logs a warning and
        continues so CLI-backed operations can still work after a herdr change.

        Raises:
            HerdrError: socket unreachable, malformed status, or stopped server.
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
        server = status.get("server")
        if not isinstance(server, dict):
            raise HerdrError("herdr status returned invalid server object")
        if not server.get("running"):
            raise HerdrError("herdr server is not running")
        proto = server.get("protocol")
        cli_server_compatible = server.get("compatible")
        is_supported_protocol = isinstance(proto, int) and not isinstance(proto, bool)
        is_supported_protocol = (
            is_supported_protocol and proto in HERDR_SUPPORTED_PROTOCOLS
        )
        if cli_server_compatible is False:
            raise HerdrProtocolError(
                "Herdr client and server protocols are incompatible; restart Herdr"
            )
        if not is_supported_protocol:
            logger.warning(
                "herdr protocol is unverified; continuing",
                server_protocol=proto,
                supported_protocols=sorted(HERDR_SUPPORTED_PROTOCOLS),
                cli_server_compatible=cli_server_compatible,
            )

    async def _agent_list_snapshot(self) -> list[HerdrLiveRecord]:
        """Read and parse one fresh ``agent.list`` snapshot.

        Sessionless records are deliberately omitted. No focus, title, name,
        directory, screen, or layout field participates in this snapshot.
        """
        result = await self._call_json(["agent", "list"])
        if result is None:
            raise HerdrAgentListError("herdr agent.list failed")
        agents = result.get("agents")
        if not isinstance(agents, list):
            raise HerdrMalformedRecordError("agent.list returned no agents list")
        records: list[HerdrLiveRecord] = []
        for agent in agents:
            if not isinstance(agent, Mapping):
                raise HerdrMalformedRecordError(
                    "agent.list contains a malformed record"
                )
            parsed = _parse_live_record(agent)
            if parsed is not None:
                records.append(parsed)
        return records

    def target_id_for_live_record(self, record: Mapping[str, object]) -> str | None:
        """Return a guarded opaque target for one ``agent.list`` record.

        Hook-side discovery uses this parser after it has established a unique
        live locator match. Malformed, sessionless, or incomplete records do
        not yield an identity.
        """
        try:
            live = _parse_live_record(record)
        except HerdrMalformedRecordError:
            return None
        return live.target_id if live is not None else None

    async def guard_session_target(self, target_id: str) -> HerdrLiveRecord:
        """Resolve one exact target against one fresh live session record."""
        if not is_herdr_session_target(target_id):
            raise HerdrUnresolvedTargetError(
                f"herdr session target has invalid format: {target_id}"
            )
        records = await self._agent_list_snapshot()
        matches = [record for record in records if record.target_id == target_id]
        if not matches:
            raise HerdrUnresolvedTargetError(
                f"herdr session target unresolved: {target_id}"
            )
        if len(matches) != 1:
            raise HerdrAmbiguousTargetError(
                f"herdr session target ambiguous: {target_id}"
            )
        return matches[0]

    @staticmethod
    def _live_ref(record: HerdrLiveRecord, label: str) -> WindowRef:
        """Project a guarded session record without exposing its locator."""
        return WindowRef(
            window_id=record.target_id,
            window_name=label,
            cwd="",
            pane_current_command=record.composite.agent,
        )

    async def _reconciliation_labels(
        self, records: Sequence[HerdrLiveRecord]
    ) -> dict[tuple[str, str], tuple[str, str, str]]:
        """Resolve safe display labels for live locators without using them as identity."""
        workspace_result = await self._call_json(["workspace", "list"])
        tab_result = await self._call_json(["tab", "list"])
        if workspace_result is None or tab_result is None:
            raise HerdrError("Herdr labels unavailable during reconciliation")
        workspace_labels = {
            workspace.get("workspace_id"): workspace.get("label")
            for workspace in workspace_result.get("workspaces", [])
            if isinstance(workspace, Mapping)
            and isinstance(workspace.get("workspace_id"), str)
            and isinstance(workspace.get("label"), str)
        }
        tab_labels = {
            tab.get("tab_id"): tab.get("label")
            for tab in tab_result.get("tabs", [])
            if isinstance(tab, Mapping)
            and isinstance(tab.get("tab_id"), str)
            and isinstance(tab.get("label"), str)
        }
        labels: dict[tuple[str, str], tuple[str, str, str]] = {}
        for record in records:
            workspace_label = workspace_labels.get(record.workspace_id)
            tab_label = tab_labels.get(record.tab_id)
            if workspace_label is None or tab_label is None:
                raise HerdrError("Herdr live locator has no display label")
            labels[(record.workspace_id, record.tab_id)] = (
                workspace_label,
                tab_label,
                format_agent_topic_prefix(workspace_label, tab_label),
            )
        return labels

    async def list_windows(self) -> list[WindowRef]:
        """List reconcilable sessionful agents keyed by opaque session targets."""
        return await self.list_windows_for_reconciliation() or []

    async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
        try:
            records = await self._agent_list_snapshot()
            labels = await self._reconciliation_labels(records)
            return [
                self._live_ref(record, labels[(record.workspace_id, record.tab_id)][2])
                for record in records
                if not _INTERNAL_LABEL_RE.match(
                    labels[(record.workspace_id, record.tab_id)][0]
                )
                and not _INTERNAL_LABEL_RE.match(
                    labels[(record.workspace_id, record.tab_id)][1]
                )
            ]
        except HerdrError:
            return None

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """Resolve a topic target through a fresh session snapshot."""
        try:
            record = await self.guard_session_target(window_id)
            labels = await self._reconciliation_labels([record])
            return self._live_ref(
                record, labels[(record.workspace_id, record.tab_id)][2]
            )
        except HerdrError:
            return None

    # ── Guarded locator operations (private) ───────────────────────────
    # These receive a locator only from the fresh session guard above.

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
        layout = result.get("layout") if result else None
        if not isinstance(layout, Mapping):
            return None
        panes = layout.get("panes")
        if isinstance(panes, Sequence) and not isinstance(panes, (str, bytes)):
            for pane in panes:
                if not isinstance(pane, Mapping) or pane.get("pane_id") != pane_id:
                    continue
                rect = pane.get("rect")
                if not isinstance(rect, Mapping):
                    continue
                width, height = rect.get("width"), rect.get("height")
                if isinstance(width, int) and isinstance(height, int):
                    return PaneDims(width=width, height=height)
        area = layout.get("area")
        if not isinstance(area, Mapping):
            return None
        width, height = area.get("width"), area.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return PaneDims(width=width, height=height)
        return None

    async def _foreground_for_pane(self, pane_id: str) -> ForegroundInfo | None:
        """Return foreground process info for a resolved pane id."""
        result = await self._call_json(["pane", "process-info", "--pane", pane_id])
        info = result.get("process_info") if result else None
        if not isinstance(info, Mapping):
            return None
        procs = info.get("foreground_processes")
        if not isinstance(procs, Sequence) or isinstance(procs, (str, bytes)):
            return None
        processes = [proc for proc in procs if isinstance(proc, Mapping)]
        if not processes:
            return None
        pgid = info.get("foreground_process_group_id")
        if not isinstance(pgid, int):
            return None
        leader = next(
            (proc for proc in processes if proc.get("pid") == pgid), processes[0]
        )
        pid = leader.get("pid")
        argv = leader.get("argv")
        cwd = leader.get("cwd")
        if (
            not isinstance(pid, int)
            or not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes))
            or not all(isinstance(arg, str) for arg in argv)
            or not isinstance(cwd, str)
        ):
            return None
        return ForegroundInfo(pid=pid, pgid=pgid, argv=list(argv), cwd=cwd, tty="")

    # ── Tab-keyed public ops (resolve tab→active-pane first) ───────────

    async def _after_action_failure(self, target_id: str) -> None:
        """Record the unavoidable post-guard dispatch race with one refresh.

        A session can move or disappear after ``guard_session_target`` and
        before herdr dispatches.  We never retarget; this refresh is solely a
        fresh observation for diagnostics/reconciliation.
        """
        with contextlib.suppress(HerdrError):
            await self.guard_session_target(target_id)

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return None
        text = await self._read_visible_pane(record.pane_id, ansi=with_ansi)
        if text is None:
            await self._after_action_failure(window_id)
        return text

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return None
        effective = min(lines, self.capabilities.read_max_lines or lines)
        text = await self._read_recent_pane(record.pane_id, lines=effective)
        if text is None:
            await self._after_action_failure(window_id)
            return None
        return CaptureResult(text=text, truncated=effective != lines)

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return None
        dims = await self._dims_for_pane(record.pane_id)
        if dims is None:
            await self._after_action_failure(window_id)
        return dims

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,
    ) -> bool:
        del raw
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return False
        ok = await self._send_to(record.pane_id, text, enter=enter, literal=literal)
        if not ok:
            await self._after_action_failure(window_id)
        return ok

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
    ) -> bool:
        """Reject raw Herdr pane IDs; only a session target may authorize I/O."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw Herdr pane operation")
            return False
        return await self.send(window_id, text, enter=enter, literal=literal)

    async def _send_to(
        self, pane_id: str, text: str, *, enter: bool, literal: bool
    ) -> bool:
        if not literal:
            keys = [_KEY_ALIASES.get(tok, tok) for tok in text.split() if tok]
            if enter:
                keys.append("Enter")
            return bool(keys) and await self._call_ok(
                ["pane", "send-keys", pane_id, *keys]
            )
        return await self._call_ok(
            ["pane", "run" if enter else "send-text", pane_id, text]
        )

    async def kill_window(self, window_id: str) -> bool:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return False
        # A tab may host multiple independently guarded agent sessions. Closing
        # the tab would terminate sibling sessions, so close only the pane
        # freshly resolved for this target.
        ok = await self._call_ok(["pane", "close", record.pane_id])
        if not ok:
            await self._after_action_failure(window_id)
        return ok

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return False
        ok = await self._call_ok(["tab", "rename", record.tab_id, new_name])
        if not ok:
            await self._after_action_failure(window_id)
        return ok

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """Return no pane handles until Herdr exposes durable sibling targets.

        The neutral ``PaneInfo.pane_id`` is actionable through pane-level APIs.
        Herdr raw pane locators are deliberately not returned across the adapter
        boundary, so a synthetic or transient ID would be misleading.
        """
        del window_id
        return []

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return
        ok = await self._call_ok(
            [
                "pane",
                "report-metadata",
                record.pane_id,
                "--source",
                "ccgram",
                "--title",
                f"ccgram:{provider_name}",
            ]
        )
        if not ok:
            await self._after_action_failure(window_id)

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return None
        value = await self._foreground_for_pane(record.pane_id)
        if value is None:
            await self._after_action_failure(window_id)
        return value

    async def agent_status(self, window_id: str) -> AgentStatus | None:
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return None
        pane = await self._pane_get(record.pane_id)
        if pane is None:
            await self._after_action_failure(window_id)
            return None
        raw_state = pane.get("agent_status")
        raw_custom_status = pane.get("custom_status")
        if raw_state is not None and not isinstance(raw_state, str):
            return None
        if raw_custom_status is not None and not isinstance(raw_custom_status, str):
            return None
        state = (raw_state or "").strip()
        return (
            AgentStatus(
                state=state,
                agent=record.composite.agent,
                custom_status=(raw_custom_status or "").strip(),
            )
            if state
            else None
        )

    async def split_window(self, window_id: str) -> str | None:
        """Return None: Herdr cannot expose an unguarded sibling pane handle.

        The neutral split contract returns a pane handle that callers can use.
        Herdr's newly allocated pane has no durable session target until an
        agent reports one, so returning its raw locator would bypass the guard.
        """
        del window_id
        return None

    async def _resolve_event_targets(
        self, window_ids: Sequence[str]
    ) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
        """Resolve event subscriptions and tab closures through fresh guards."""
        pane_to_target: dict[str, str] = {}
        tab_to_targets: dict[str, list[str]] = {}
        for target_id in window_ids:
            try:
                record = await self.guard_session_target(target_id)
            except HerdrError:
                continue
            pane_to_target[record.pane_id] = target_id
            tab_to_targets.setdefault(record.tab_id, []).append(target_id)
        return pane_to_target, {
            tab_id: tuple(targets) for tab_id, targets in tab_to_targets.items()
        }

    async def _resolve_panes(self, window_ids: Sequence[str]) -> dict[str, str]:
        """Compatibility helper returning only pane-to-target subscriptions."""
        panes, _tabs = await self._resolve_event_targets(window_ids)
        return panes

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """List all herdr workspaces as neutral ``WorkspaceRef`` objects.

        Returns ``[]`` when the workspace command is unavailable (older herdr
        server) — callers must handle the empty case gracefully (fall through
        to cwd-resolve).
        """
        result = await self._call_json(["workspace", "list"])
        workspaces = result.get("workspaces") if result else None
        if not isinstance(workspaces, list):
            return []
        panes: list[Mapping[str, object]] | None = None
        refs: list[WorkspaceRef] = []
        for workspace in workspaces:
            if not isinstance(workspace, Mapping):
                return []
            workspace_id = workspace.get("workspace_id")
            label = workspace.get("label")
            cwd = workspace.get("cwd")
            if not (
                isinstance(workspace_id, str)
                and workspace_id
                and isinstance(label, str)
            ):
                return []
            if not isinstance(cwd, str):
                if panes is None:
                    pane_result = await self._call_json(["pane", "list"])
                    raw_panes = pane_result.get("panes") if pane_result else None
                    if not isinstance(raw_panes, list) or not all(
                        isinstance(pane, Mapping) for pane in raw_panes
                    ):
                        return []
                    panes = raw_panes
                cwd = _workspace_cwd_from_panes(workspace, panes)
                if cwd is None:
                    return []
            refs.append(WorkspaceRef(workspace_id, label, cwd))
        return refs

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
        """Compatibility creation API that never returns a Herdr tab binding.

        A sessionful launch without a picker selection creates a workspace at
        *work_dir* explicitly and uses its returned opaque ID. Tmux retains its
        existing behavior through its own implementation.
        """
        if not start_agent or not launch_command:
            return (
                False,
                "Herdr topic creation requires a sessionful agent",
                "",
                "",
            )
        try:
            target = await self.create_topic_target(
                work_dir,
                launch_command=launch_command,
                workspace_id=workspace_id,
                window_name=window_name,
                agent_args=agent_args,
            )
        except HerdrError as exc:
            return False, str(exc), "", ""
        return (
            True,
            f"Created Herdr session target '{target.label}'",
            target.label,
            target.target_id,
        )

    async def _await_agent_on_pane(self, pane_id: str) -> None:
        """Wait until Herdr identifies the newly launched agent pane."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CREATED_SESSION_DISCOVERY_TIMEOUT_SECONDS
        while True:
            result = await self._call_json(["agent", "list"])
            agents = result.get("agents") if result else None
            if isinstance(agents, list) and any(
                isinstance(agent, Mapping) and agent.get("pane_id") == pane_id
                for agent in agents
            ):
                return
            if loop.time() >= deadline:
                break
            await asyncio.sleep(_CREATED_SESSION_POLL_INTERVAL_SECONDS)
        raise HerdrUnresolvedTargetError("new Herdr pane did not identify an agent")

    async def _await_created_session_target(
        self,
        *,
        tab_id: str,
        pane_id: str,
        workspace_id: str | None,
    ) -> HerdrLiveRecord:
        """Wait for exactly one session reported for a newly-created pane."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CREATED_SESSION_DISCOVERY_TIMEOUT_SECONDS
        while True:
            matches = [
                record
                for record in await self._agent_list_snapshot()
                if record.tab_id == tab_id
                and record.pane_id == pane_id
                and (workspace_id is None or record.workspace_id == workspace_id)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise HerdrAmbiguousTargetError(
                    "new Herdr pane reported duplicate sessions"
                )
            if loop.time() >= deadline:
                break
            await asyncio.sleep(_CREATED_SESSION_POLL_INTERVAL_SECONDS)
        raise HerdrUnresolvedTargetError("new Herdr pane did not report a session")

    async def create_topic_target(  # noqa: C901
        self,
        work_dir: str,
        *,
        launch_command: str | None,
        workspace_id: str | None,
        window_name: str | None = None,
        agent_args: str = "",
        initial_input: str | None = None,
    ) -> TopicTargetResult:
        """Create an agent tab and return its guarded session target.

        A picker-selected workspace is validated exactly. Without a selection,
        this transaction creates a workspace at *work_dir* and uses only its
        returned ID; it never infers an active or matching workspace. Herdr
        locators are used only during this transaction. A failed launch,
        missing report, or duplicate report closes the newly-created tab; it
        never closes a picker-selected workspace.
        """
        path = Path(work_dir).expanduser()
        if not path.is_dir():
            raise HerdrError(f"Directory does not exist: {work_dir}")
        owned_workspace_id: str | None = None
        if workspace_id:
            workspaces = await self.list_workspaces()
            if workspace_id not in {workspace.workspace_id for workspace in workspaces}:
                raise HerdrError("Selected Herdr workspace no longer exists")
        else:
            created_workspace = await self._call_json(
                ["workspace", "create", "--cwd", str(path), "--no-focus"]
            )
            workspace = (created_workspace or {}).get("workspace")
            workspace_id = (
                workspace.get("workspace_id")
                if isinstance(workspace, Mapping)
                else None
            )
            if not isinstance(workspace_id, str) or not workspace_id:
                raise HerdrError("herdr workspace creation returned no workspace id")
            owned_workspace_id = workspace_id

        tab_id: str | None = None
        try:
            args = [
                "tab",
                "create",
                "--cwd",
                str(path),
                "--no-focus",
                "--workspace",
                workspace_id,
            ]
            if window_name:
                args += ["--label", window_name]
            result = await self._call_json(args)
            tab = (result or {}).get("tab") or {}
            root = (result or {}).get("root_pane") or {}
            tab_id = tab.get("tab_id") if isinstance(tab, Mapping) else None
            pane_id = root.get("pane_id") if isinstance(root, Mapping) else None
            if not isinstance(tab_id, str) or not tab_id:
                raise HerdrError("herdr tab creation returned no tab id")
            # A tab may have been allocated even when the response omitted its
            # root pane. Close it before closing the workspace we created.
            if not isinstance(pane_id, str) or not pane_id:
                raise HerdrError("herdr tab creation returned no root pane")
            if launch_command:
                command = f"{launch_command} {agent_args}".strip()
                if not await self._call_ok(["pane", "run", pane_id, command]):
                    raise HerdrError("Failed to start agent in Herdr tab")
            if initial_input is not None:
                await self._await_agent_on_pane(pane_id)
                if not await self._call_ok(["pane", "run", pane_id, initial_input]):
                    raise HerdrError("Failed to send initial agent message")
            record = await self._await_created_session_target(
                tab_id=tab_id, pane_id=pane_id, workspace_id=workspace_id
            )
            return TopicTargetResult(
                record.target_id,
                tab.get("label", window_name or ""),
                tab_id,
                pane_id,
            )
        except BaseException:
            if tab_id:
                await self._call_ok(["tab", "close", tab_id])
            if owned_workspace_id:
                await self._call_ok(["workspace", "close", owned_workspace_id])
            raise

    async def create_worktree_window(  # noqa: C901, PLR0911
        self,
        repo_path: str,
        worktree_path: str,
        branch: str,
        *,
        window_name: str | None = None,
        launch_command: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Delegate worktree creation to herdr (``worktree create``).

        One ``worktree create`` makes the git checkout at *worktree_path* on
        *branch* (off the repo at *repo_path*), opens it as a herdr
        workspace+tab grouped under the parent repo, and returns a topic-safe
        opaque agent-session target. We then ``pane run`` *launch_command* in
        the root pane and wait for that exact pane to report its session.
        ``window_id`` in the legacy tuple is therefore the durable target, not
        the transient tab locator.
        """
        repo = Path(repo_path).expanduser()
        if not repo.is_dir():
            return False, f"Repo path is not a directory: {repo_path}", "", ""

        args = [
            "worktree",
            "create",
            "--cwd",
            str(repo),
            "--branch",
            branch,
            "--path",
            worktree_path,
            "--no-focus",
            "--json",
        ]
        if window_name:
            args += ["--label", window_name]
        result = await self._call_json(args)
        if not result:
            return False, f"Failed to create herdr worktree at {worktree_path}", "", ""

        tab = result.get("tab") or {}
        root_pane = result.get("root_pane") or {}
        workspace = result.get("workspace") or {}
        if not all(isinstance(value, Mapping) for value in (tab, root_pane, workspace)):
            return False, "herdr worktree returned malformed creation data", "", ""
        # tab_id from tab/root_pane; fall back to the new workspace's active tab.
        tab_id = tab.get("tab_id") or root_pane.get("tab_id", "")
        if not tab_id:
            tab_id = workspace.get("active_tab_id", "")
        pane_id = root_pane.get("pane_id")
        if not isinstance(tab_id, str) or not tab_id:
            return False, "herdr worktree created without a tab id", "", ""
        if not isinstance(pane_id, str) or not pane_id:
            # The worktree exists, but it is unsafe to bind a topic without a
            # specific pane/session. Close only the new tab, never the workspace.
            await self._call_ok(["tab", "close", tab_id])
            return False, "herdr worktree created without a root pane", "", ""
        label = tab.get("label", window_name or "")
        if not isinstance(label, str):
            await self._call_ok(["tab", "close", tab_id])
            return False, "herdr worktree created without a valid tab label", "", ""
        created_workspace = workspace.get("workspace_id")
        if created_workspace is not None and not isinstance(created_workspace, str):
            await self._call_ok(["tab", "close", tab_id])
            return False, "herdr worktree created without a valid workspace id", "", ""
        workspace_id = created_workspace

        try:
            if launch_command and not await self._call_ok(
                ["pane", "run", pane_id, launch_command]
            ):
                raise HerdrError("Failed to start agent in Herdr worktree")
            record = await self._await_created_session_target(
                tab_id=tab_id, pane_id=pane_id, workspace_id=workspace_id
            )
        except BaseException as exc:
            await self._call_ok(["tab", "close", tab_id])
            if isinstance(exc, HerdrError):
                return False, str(exc), "", ""
            raise

        logger.info("Created herdr worktree target %r at %s", label, worktree_path)
        return (
            True,
            f"Created herdr worktree '{branch}' at {worktree_path}",
            label,
            record.target_id,
        )

    async def watch_events(  # noqa: C901
        self, window_ids: Sequence[str]
    ) -> AsyncGenerator[MuxEvent, None]:
        """Stream push events for *window_ids* (see ``Multiplexer.watch_events``).

        Subscribes to global ``tab.closed`` plus per-pane
        ``pane.agent_status_changed`` for the active panes of *window_ids*
        (agent-status subscriptions require a pane id). Reprimes each pane's
        current status once the subscription is live (on the ``SUBSCRIBED``
        sentinel, so a status change during reprime is buffered, not lost), then
        yields translated events until the stream drops and reconnects with
        backoff. Cancelling the iterator closes the socket. The watched set is
        fixed per call: herdr cannot add subscriptions to a live connection, so
        the consumer restarts this iterator with a new set when bindings change.
        """
        ids = list(window_ids)
        backoff = _STREAM_BACKOFF_BASE
        while True:
            pane_to_window, tab_to_windows = await self._resolve_event_targets(ids)
            subscriptions: list[Mapping[str, object]] = [
                {"type": "tab.closed"},
                *(
                    subscription
                    for pane in pane_to_window
                    for subscription in (
                        {"type": "pane.agent_status_changed", "pane_id": pane},
                        {"type": "pane.exited", "pane_id": pane},
                        {"type": "pane.closed", "pane_id": pane},
                    )
                ),
            ]
            refresh_subscriptions = False
            try:
                async with contextlib.aclosing(
                    self._open_stream(subscriptions)
                ) as stream:
                    while True:
                        try:
                            async with asyncio.timeout(_STREAM_REPRIME_INTERVAL):
                                obj = await anext(stream)
                        except TimeoutError:
                            # No event may arrive after a target moves because
                            # Herdr subscriptions are pane-specific. Reconnect
                            # with fresh guarded locators instead of waiting for
                            # an event on the stale pane forever.
                            refresh_subscriptions = True
                            break
                        if is_subscribed_sentinel(obj):
                            # Subscription is live — reprime now so the status cache
                            # isn't cold; events during reprime are buffered + read
                            # on the next iterations (no reprime-vs-subscribe race).
                            backoff = _STREAM_BACKOFF_BASE
                            for pane_id, window_id in pane_to_window.items():
                                status = await self.agent_status(window_id)
                                if status is not None:
                                    yield MuxEvent(
                                        kind="agent_status",
                                        window_id=window_id,
                                        pane_id=pane_id,
                                        status=status,
                                    )
                            continue
                        # Terminal events identify the pane/tab that just vanished.
                        # Resolve and emit them through the pre-refresh guard: a
                        # fresh snapshot cannot contain the closed locator, so
                        # refreshing first would silently drop the close event.
                        guarded_terminal_events = tuple(
                            event
                            for event in translate_event(
                                obj, pane_to_window, tab_to_windows
                            )
                            if event.kind == "window_died"
                        )
                        if guarded_terminal_events:
                            for event in guarded_terminal_events:
                                yield event
                            continue
                        # Agent locators can move while a stream is open. Herdr does
                        # not support incremental subscription updates, so refresh
                        # the guarded mapping and reconnect before translating status
                        # events whenever a move is observed.
                        fresh_panes, fresh_tabs = await self._resolve_event_targets(ids)
                        if (
                            fresh_panes != pane_to_window
                            or fresh_tabs != tab_to_windows
                        ):
                            refresh_subscriptions = True
                            break
                        for event in translate_event(
                            obj, pane_to_window, tab_to_windows
                        ):
                            yield event
            except OSError as exc:
                logger.debug("herdr event stream error: %s", exc)
            if refresh_subscriptions:
                # A mapping change is a healthy re-subscription, not a transport
                # failure; do not penalize it with exponential backoff.
                continue
            # Clean EOF or socket error → back off, then reconnect with the full
            # set (incremental subscribe is unsupported) and reprime.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _STREAM_BACKOFF_MAX)

    # ── Transitional surface (remaining legacy helpers) ───────────────
    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,
    ) -> str | None:
        """Capture only when the supplied value is the guarded session target."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw Herdr pane capture")
            return None
        return await self.capture_pane(window_id, with_ansi=with_ansi)

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
        """Return a guarded target's pane title."""
        try:
            record = await self.guard_session_target(window_id)
        except HerdrError:
            return ""
        pane = await self._pane_get(record.pane_id)
        if pane is None:
            await self._after_action_failure(window_id)
            return ""
        return pane.get("title", "") or ""
