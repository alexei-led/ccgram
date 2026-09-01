"""Herdr backend for the Multiplexer contract, via the herdr CLI/socket.

Anti-corruption layer over `herdr <https://github.com/ogulcancelik/herdr>`_'s
Unix-socket JSON-RPC CLI. Every herdr JSON shape (``pane_info`` / ``pane_list``
/ ``pane_process_info`` / ``pane_layout`` / ``tab_created`` …) and every
``wN:pN``/``wN:tN`` id string stays **private** to this module; callers see
only the neutral value types from ``multiplexer.base`` (design "Module map":
herdr.py is adapter, anti-corruption).

Identity mapping: Herdr ``agent.list`` is the sole identity source. A complete
agent-session composite becomes an opaque durable target. Older Herdr records
for known hook-capable agents may omit ``agent_session``; those records use the
current terminal identity as a short-lived fallback so hooks and reconciliation
can still agree on a live target. Raw locators are used only after a fresh guard
authorizes one action; they are never persisted as aliases.

The backend shells out to the ``herdr`` CLI (which the design explicitly allows
as an alternative to talking the socket directly); the socket path is passed
through ``$HERDR_SOCKET_PATH``. The command runner is injectable so unit tests
feed JSON fixtures without a live socket and the constructor stays I/O-free
(the proxy/registry can build the backend before bootstrap; the socket is only
touched on the first real call).

Capabilities (design "MultiplexerCapabilities"): ``ids_stable_across_restart``
is False (a herdr *server* restart re-mints ids, and nothing re-resolves a
stale target: guarded targets hash the agent-session composite, so a
persisted one simply stops matching), ``exposes_pane_tty`` is False (no tty in ``process-info`` on
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

from ..herdr_targets import HERDR_SESSION_TARGET_PREFIX, is_herdr_session_target
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
# 14–20 are supported. Other versions are attempted with a warning so ccgram
# remains usable across Herdr upgrades and downgrades.
HERDR_SUPPORTED_PROTOCOLS = frozenset(range(14, 21))
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
    supports_display_name_rebind=False,
    supports_workspace_selection=True,
    native_topic_targets=True,
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

# Agent TUIs (Claude Code, Codex, Pi) read a submit key that arrives in the
# same input batch as the prompt text as a literal newline, so the prompt is
# typed but never sent. ``pane run`` delivers exactly that batch, so a literal
# submit is split into ``send-text`` + a separate ``Enter``, with this gap for
# the TUI to consume the text first. Mirrors the tmux backend's 0.5s delay in
# ``_send_literal_then_enter``.
_SEND_ENTER_DELAY_SECONDS = 0.5

# Event-stream reconnect backoff (seconds): exponential, capped.
_STREAM_BACKOFF_BASE = 1.0
_STREAM_BACKOFF_MAX = 30.0
# A live stream has no locator-change notification. Re-prime periodically so a
# target that moved to another pane receives a fresh per-pane subscription.
_STREAM_REPRIME_INTERVAL = 5.0


def _workspace_cwd_from_panes(
    workspace: Mapping[str, object], panes: Sequence[Mapping[str, object]]
) -> str | None:
    """Return the active tab's shared stable CWD from a protocol-19 snapshot."""
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

    def shared_cwd(field: str) -> str | None:
        cwd: str | None = None
        for pane in candidates:
            value = pane.get(field)
            if not isinstance(value, str) or not value:
                return None
            if cwd is None:
                cwd = value
            elif cwd != value:
                return None
        return cwd

    has_stable_cwd = any(
        isinstance(pane.get("cwd"), str) and pane.get("cwd") for pane in candidates
    )
    return shared_cwd("cwd") if has_stable_cwd else shared_cwd("foreground_cwd")


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
    """The complete input for an opaque Herdr target identity."""

    source: str
    agent: str
    kind: str
    value: str


@dataclass(frozen=True)
class HerdrLiveRecord:
    """One detected agent and its short-lived current Herdr locator."""

    target_id: str
    composite: HerdrSessionComposite
    terminal_id: str
    pane_id: str
    tab_id: str
    workspace_id: str
    cwd: str = ""


def _session_field(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _session_composite(record: Mapping[str, object]) -> HerdrSessionComposite | None:
    """Parse one complete ``agent_session`` value, if Herdr published one."""
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
    # Field order is part of the persisted target-ID protocol. A golden test
    # pins it so refactors cannot silently orphan existing topic bindings.
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return payload.encode("utf-8")


def herdr_session_target_id(composite: HerdrSessionComposite) -> str:
    """Return the opaque versioned ID for a complete session composite."""
    prefix = b"ccgram-herdr-session-v1\0"
    digest = hashlib.sha256(prefix + canonical_session_bytes(composite)).hexdigest()
    return f"{HERDR_SESSION_TARGET_PREFIX}{digest}"


def _parse_live_record(record: Mapping[str, object]) -> HerdrLiveRecord | None:
    composite = _session_composite(record)
    locators = {
        key: _session_field(record.get(key))
        for key in ("terminal_id", "pane_id", "tab_id", "workspace_id")
    }
    if any(value is None for value in locators.values()):
        raise HerdrMalformedRecordError(
            "agent.list contains an incomplete live locator"
        )
    if composite is None:
        agent = _session_field(record.get("agent"))
        terminal_id = locators["terminal_id"]
        if agent not in {"claude", "codex", "gemini"} or terminal_id is None:
            return None
        # Providers that may remain sessionless still expose a unique terminal
        # identity. Pi is excluded: Herdr publishes its durable session shortly
        # after startup, and creating a terminal topic in that gap would create
        # a second topic when the durable identity arrives.
        composite = HerdrSessionComposite("herdr", agent, "terminal", terminal_id)
    target_id = herdr_session_target_id(composite)
    # ``cwd`` is the agent's own working directory; ``foreground_cwd`` follows
    # whatever the agent currently shells into (a worktree, a plugin cache) and
    # would send hookless transcript discovery to the wrong session directory.
    cwd = _session_field(record.get("cwd")) or ""
    return HerdrLiveRecord(
        target_id=target_id,
        composite=composite,
        terminal_id=locators["terminal_id"] or "",
        pane_id=locators["pane_id"] or "",
        tab_id=locators["tab_id"] or "",
        workspace_id=locators["workspace_id"] or "",
        cwd=cwd,
    )


def _parse_agent_records(
    agents: Sequence[object],
) -> tuple[list[HerdrLiveRecord], int]:
    """Parse the records, counting the ones that leave a gap in the account.

    The count is malformed records only. A record that parses to None is not a
    gap: a recognised agent without a composite already takes a terminal-derived
    identity, so what remains is an agent ccgram cannot address at all and never
    holds a binding to.
    """
    records: list[HerdrLiveRecord] = []
    malformed = 0
    for agent in agents:
        if not isinstance(agent, Mapping):
            malformed += 1
            continue
        try:
            parsed = _parse_live_record(agent)
        except HerdrMalformedRecordError:
            malformed += 1
            continue
        if parsed is not None:
            records.append(parsed)
    return records, malformed


def _quarantine_ambiguous_records(
    records: Sequence[HerdrLiveRecord],
) -> tuple[list[HerdrLiveRecord], int, int]:
    target_counts: dict[str, int] = {}
    pane_counts: dict[str, int] = {}
    for record in records:
        target_counts[record.target_id] = target_counts.get(record.target_id, 0) + 1
        pane_counts[record.pane_id] = pane_counts.get(record.pane_id, 0) + 1
    duplicate_targets = {
        target_id for target_id, count in target_counts.items() if count > 1
    }
    duplicate_panes = {pane_id for pane_id, count in pane_counts.items() if count > 1}
    safe = [
        record
        for record in records
        if record.target_id not in duplicate_targets
        and record.pane_id not in duplicate_panes
    ]
    return safe, len(duplicate_targets), len(duplicate_panes)


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
        """Verify Herdr is reachable; warn but do not gate on compatibility.

        ``HERDR_SUPPORTED_PROTOCOLS`` are accepted without a warning. Other
        protocol versions and a false CLI compatibility flag are best-effort:
        ccgram logs a warning and continues so CLI-backed operations can try
        the current command surface after a Herdr change. Individual commands
        still report their own transport or schema failures.

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
        if not is_supported_protocol or cli_server_compatible is False:
            logger.warning(
                "herdr protocol is unverified; continuing",
                server_protocol=proto,
                supported_protocols=sorted(HERDR_SUPPORTED_PROTOCOLS),
                cli_server_compatible=cli_server_compatible,
            )

    async def _agent_list_snapshot(self) -> list[HerdrLiveRecord]:
        """The safe records from one fresh ``agent.list`` snapshot.

        Known hook-capable agents without a complete session composite use a
        terminal fallback identity. Shells and malformed records remain
        ignored. No focus, title, name, directory, screen, or layout field
        participates in this snapshot.

        Quarantined records are dropped silently, so this subset is safe to
        address and safe to show, but it is not a complete account of what is
        live. Anything answering "does this window still exist" must take
        ``_agent_list_snapshot_checked`` instead.
        """
        records, _complete = await self._agent_list_snapshot_checked()
        return records

    async def _agent_list_snapshot_checked(
        self,
    ) -> tuple[list[HerdrLiveRecord], bool]:
        """The safe records, plus whether they account for every live agent.

        ``False`` when malformed or ambiguous records were quarantined: those
        panes are live, ccgram may hold bindings to them, and they are absent
        from the subset returned here. Reporting that subset as complete makes
        such a binding look like a ghost, which the audit, the polling loop and
        the destructive guards all act on.

        Completeness means every *addressable* guarded target. A record that
        merely names a tool is not one: a recognised agent without a composite
        already gets a terminal-derived identity, and what is left parses to
        None because ccgram does not recognise the agent at all. Antigravity is
        the live example, supported here and documented to stay sessionless
        until its first prompt, so counting it would make reconciliation
        unknown for every topic while one sits idle.
        """
        result = await self._call_json(["agent", "list"])
        if result is None:
            raise HerdrAgentListError("herdr agent.list failed")
        agents = result.get("agents")
        if not isinstance(agents, list):
            raise HerdrMalformedRecordError("agent.list returned no agents list")
        records, malformed = _parse_agent_records(agents)
        if malformed:
            logger.warning(
                "quarantining malformed Herdr agent records",
                malformed_record_count=malformed,
            )
        safe, duplicate_targets, duplicate_panes = _quarantine_ambiguous_records(
            records
        )
        if duplicate_targets or duplicate_panes:
            logger.warning(
                "quarantining ambiguous Herdr agent records",
                duplicate_target_count=duplicate_targets,
                duplicate_pane_count=duplicate_panes,
            )
        complete = not (malformed or duplicate_targets or duplicate_panes)
        return safe, complete

    def target_id_for_live_record(self, record: Mapping[str, object]) -> str | None:
        """Return an opaque target for one structurally valid live record."""
        try:
            live = _parse_live_record(record)
        except HerdrMalformedRecordError:
            return None
        return live.target_id if live is not None else None

    def target_id_for_live_snapshot(
        self,
        record: Mapping[str, object],
        snapshot: Sequence[object],
    ) -> str | None:
        """Return a target only when the full snapshot is unambiguous."""
        try:
            selected = _parse_live_record(record)
        except HerdrMalformedRecordError:
            return None
        if selected is None:
            return None
        records, malformed = _parse_agent_records(snapshot)
        safe, duplicate_targets, duplicate_panes = _quarantine_ambiguous_records(
            records
        )
        if malformed or duplicate_targets or duplicate_panes:
            logger.warning(
                "quarantining unsafe Herdr hook snapshot",
                malformed_record_count=malformed,
                duplicate_target_count=duplicate_targets,
                duplicate_pane_count=duplicate_panes,
            )
        return selected.target_id if selected in safe else None

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
    def _live_ref(
        record: HerdrLiveRecord, label: str, *, adoptable: bool = True
    ) -> WindowRef:
        """Project a live record without exposing reusable locator aliases.

        ``topic_eligible`` is stamped here because this parser is already the
        place that decides what counts: it emits a record only for a live agent
        carrying a guarded target, and a bare shell pane never reaches it. The
        verdict travels on the window so discovery needs no herdr-shaped check
        of its own.
        """
        return WindowRef(
            window_id=record.target_id,
            window_name=label,
            cwd=record.cwd,
            pane_current_command=record.composite.agent,
            topic_eligible=adoptable
            and is_herdr_session_target(record.target_id)
            and bool(record.composite.agent.strip()),
        )

    async def _reconciliation_labels(
        self, records: Sequence[HerdrLiveRecord]
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Resolve best-effort display labels without using them as identity."""
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
        labels: dict[tuple[str, str], tuple[str, str]] = {}
        missing = 0
        for record in records:
            workspace_label = workspace_labels.get(record.workspace_id)
            tab_label = tab_labels.get(record.tab_id)
            if workspace_label is None or tab_label is None:
                missing += 1
                continue
            labels[(record.workspace_id, record.tab_id)] = (
                workspace_label,
                tab_label,
            )
        if missing:
            logger.warning(
                "skipping Herdr sessions with missing display labels",
                missing_record_count=missing,
            )
        return labels

    async def _project_live_refs(
        self,
        records: Sequence[HerdrLiveRecord],
        *,
        include_internal: bool = False,
        include_unlabeled: bool = False,
    ) -> list[WindowRef]:
        """Project one agent snapshot with best-effort live labels."""
        labels = await self._reconciliation_labels(records)
        refs: list[WindowRef] = []
        for record in records:
            label = labels.get((record.workspace_id, record.tab_id))
            if label is None:
                if include_unlabeled:
                    # Kept for liveness and addressing, never for adoption:
                    # without labels this cannot tell an internal ``__*__``
                    # workspace or tab from an ordinary one, and adopting
                    # ccgram's own pane is the failure that guard exists for.
                    refs.append(
                        self._live_ref(
                            record,
                            format_agent_topic_prefix(
                                "Herdr",
                                record.target_id[-12:],
                                provider=record.composite.agent,
                            ),
                            adoptable=False,
                        )
                    )
                continue
            workspace_label, tab_label = label
            internal = bool(
                _INTERNAL_LABEL_RE.match(workspace_label)
                or _INTERNAL_LABEL_RE.match(tab_label)
            )
            if internal and not include_internal:
                continue
            pane = record.pane_id.rsplit(":", 1)[-1]
            refs.append(
                self._live_ref(
                    record,
                    format_agent_topic_prefix(
                        workspace_label,
                        tab_label,
                        pane,
                        provider=record.composite.agent,
                    ),
                    adoptable=not internal,
                )
            )
        return refs

    async def list_windows(self) -> list[WindowRef]:
        """Return a best-effort UI listing, hiding ccgram's own internal panes.

        Built from the safe subset directly, not from the reconciliation
        listing: quarantined records make that listing report unknown, and a
        picker that empties itself because one unrelated record was malformed
        would be worse than one showing everything it can address. Showing what
        is addressable is exactly this listing's job.
        """
        try:
            refs = await self._project_live_refs(await self._agent_list_snapshot())
        except HerdrError:
            # Best-effort, as before: a failed agent, workspace or tab listing
            # empties the picker instead of raising through it. Only the
            # reconciliation listing distinguishes this from "nothing there".
            return []
        return [w for w in refs if w.topic_eligible]

    async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
        """Every live record, including internal ones, marked unadoptable.

        Internal ``__*__`` workspaces and tabs are ccgram's own and must never
        be adopted, but they have to stay in the listing: a bound session
        renamed into one would otherwise vanish from liveness, be audited as a
        ghost binding and have its topic closed. Existence and adoptability are
        different questions, so this answers the first and stamps the second.
        """
        try:
            records, complete = await self._agent_list_snapshot_checked()
            if not complete:
                # Quarantined records are live panes missing from this subset,
                # so it cannot answer what exists. Saying so beats handing back
                # a listing whose gaps read as dead bindings.
                logger.warning(
                    "herdr agent.list incomplete; reporting liveness unknown"
                )
                return None
            return await self._project_live_refs(
                records, include_internal=True, include_unlabeled=True
            )
        except HerdrError:
            return None

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """Resolve a topic target through one consistent live snapshot."""
        if not is_herdr_session_target(window_id):
            return None
        try:
            records = await self._agent_list_snapshot()
            matches = [record for record in records if record.target_id == window_id]
            if len(matches) != 1:
                return None
            refs = await self._project_live_refs(
                records, include_internal=True, include_unlabeled=True
            )
            return next((ref for ref in refs if ref.window_id == window_id), None)
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
        if argv is None:
            # An agent that rewrites its process title (Pi runs on node and
            # renames itself to "pi") is published with argv0 but no argv.
            # argv0 carries the identity callers classify on; ``name`` is the
            # runtime ("node") and would misclassify the pane.
            #
            # Only synthesize when argv0 really is a rename. A plain shell
            # publishes argv0 == name, and a one-element argv is exactly what
            # `shell_infra._is_interactive_shell` reads as "idle at a prompt,
            # safe to interrupt" — so faking one for `bash ./deploy.sh` whose
            # args happened to be unreadable would earn a running script a C-c.
            # Falling through to None keeps that detection fail-safe.
            argv0 = leader.get("argv0")
            name = leader.get("name")
            # A rename has to be *observed*: both fields present and different.
            # Treating an absent ``name`` as evidence of one would synthesize
            # argv for the very record shape this guard exists to reject.
            renamed = (
                isinstance(argv0, str)
                and bool(argv0)
                and isinstance(name, str)
                and argv0.rsplit("/", 1)[-1].lstrip("-") != name
            )
            argv = [argv0] if renamed else None
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
        if not await self._call_ok(["pane", "send-text", pane_id, text]):
            return False
        if not enter:
            return True
        # Never collapse back into ``pane run``: the batched Enter is the bug.
        await asyncio.sleep(_SEND_ENTER_DELAY_SECONDS)
        return await self._call_ok(["pane", "send-keys", pane_id, "Enter"])

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
            records = await self._agent_list_snapshot()
        except HerdrError:
            return False
        matches = [record for record in records if record.target_id == window_id]
        if len(matches) != 1:
            return False
        record = matches[0]
        siblings = [
            candidate
            for candidate in records
            if (candidate.workspace_id, candidate.tab_id)
            == (record.workspace_id, record.tab_id)
        ]
        if len(siblings) > 1:
            logger.warning(
                "refusing to rename shared Herdr tab through one agent topic",
                target_id=window_id,
                tab_id=record.tab_id,
                agent_count=len(siblings),
            )
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
        """Resolve all event subscriptions from one fresh agent snapshot."""
        try:
            records = await self._agent_list_snapshot()
        except HerdrError:
            return {}, {}
        requested = set(window_ids)
        pane_to_target: dict[str, str] = {}
        tab_to_targets: dict[str, list[str]] = {}
        for record in records:
            if record.target_id not in requested:
                continue
            pane_to_target[record.pane_id] = record.target_id
            tab_to_targets.setdefault(record.tab_id, []).append(record.target_id)
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
            label = tab.get("label") if isinstance(tab, Mapping) else None
            if not isinstance(tab_id, str) or not tab_id:
                raise HerdrError("herdr tab creation returned no tab id")
            if not isinstance(label, str) or not label:
                raise HerdrError("herdr tab creation returned no valid label")
            # A tab may have been allocated even when the response omitted its
            # root pane. Close it before closing the workspace we created.
            if not isinstance(pane_id, str) or not pane_id:
                raise HerdrError("herdr tab creation returned no root pane")
            if launch_command:
                command = f"{launch_command} {agent_args}".strip()
                if not await self._call_ok(["pane", "run", pane_id, command]):
                    raise HerdrError("Failed to start agent in Herdr tab")
            record = await self._await_created_session_target(
                tab_id=tab_id,
                pane_id=pane_id,
                workspace_id=workspace_id,
            )
            refs = await self._project_live_refs([record])
            if len(refs) != 1:
                raise HerdrError("new Herdr pane has no valid display metadata")
            return TopicTargetResult(
                record.target_id,
                refs[0].window_name,
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
        if not isinstance(label, str) or not label:
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
                tab_id=tab_id,
                pane_id=pane_id,
                workspace_id=workspace_id,
            )
        except BaseException as exc:
            await self._call_ok(["tab", "close", tab_id])
            if isinstance(exc, HerdrError):
                return False, str(exc), "", ""
            raise

        refs = await self._project_live_refs([record])
        if len(refs) != 1:
            await self._call_ok(["tab", "close", tab_id])
            return (
                False,
                "new Herdr worktree pane has no valid display metadata",
                "",
                "",
            )
        label = refs[0].window_name
        logger.info("Created herdr worktree target %r at %s", label, worktree_path)
        return (
            True,
            f"Created herdr worktree '{branch}' at {worktree_path}",
            label,
            record.target_id,
        )

    async def watch_events(  # noqa: C901, PLR0912
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
                    pending_event: asyncio.Task[dict] | None = None
                    try:
                        while True:
                            if pending_event is None:
                                pending_event = asyncio.create_task(anext(stream))
                            done, _ = await asyncio.wait(
                                {pending_event}, timeout=_STREAM_REPRIME_INTERVAL
                            )
                            if not done:
                                # Keep the socket read pending while checking
                                # whether pane-specific subscriptions changed.
                                (
                                    fresh_panes,
                                    fresh_tabs,
                                ) = await self._resolve_event_targets(ids)
                                if (
                                    fresh_panes != pane_to_window
                                    or fresh_tabs != tab_to_windows
                                ):
                                    refresh_subscriptions = True
                                    break
                                continue
                            try:
                                obj = pending_event.result()
                            except StopAsyncIteration:
                                pending_event = None
                                break
                            pending_event = None
                            if is_subscribed_sentinel(obj):
                                # Subscription is live — reprime now so the status
                                # cache isn't cold. Events are buffered meanwhile.
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
                            # Resolve terminal events through the pre-refresh guard:
                            # a fresh snapshot cannot contain a closed locator.
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
                            # Status events may reveal a moved agent. Reconnect before
                            # translating them if the guarded mapping changed.
                            fresh_panes, fresh_tabs = await self._resolve_event_targets(
                                ids
                            )
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
                    finally:
                        if pending_event is not None:
                            pending_event.cancel()
                            await asyncio.gather(pending_event, return_exceptions=True)
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
