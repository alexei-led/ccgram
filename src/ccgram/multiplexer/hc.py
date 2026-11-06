"""herdr-core (``hc``) backend for the Multiplexer contract.

Anti-corruption layer over herdr-core, the terminal runtime behind the Herdr
desktop app (``herdr-app``). Not to be confused with the ``herdr`` backend,
which drives the separate herdrdev/herdr terminal multiplexer. herdr-core is
reached over its line-delimited JSON-RPC 2.0 socket (``$HC_TERMINAL_SOCKET``,
default ``~/.local/state/herdr-core/default/default.sock``; see ``hc_socket``),
and tabs and spaces, which belong to the app rather than the core, through the
``herdr-app do`` command line. Every herdr-core JSON shape stays private to
this module, ``hc_socket`` and ``hc_events``; callers only see the neutral
value types from ``multiplexer.base``.

Identity: ``window_id = "hc-" + address``. herdr-core addresses (``t5``,
``t1y``; ``machine/t7`` for a terminal on a connected machine) come from a
monotonically increasing counter that survives core restarts, so an address is
never reused and ``ids_stable_across_restart`` is True. A remote address keeps
its slash; it is validated against a strict pattern before use and is never
put into a path. Remote terminals are listed for liveness but never offered or
adopted: their transcripts and hooks live on the other machine.

One terminal is one window with one pane: ``list_panes`` returns ``[]`` and
``split_window`` returns None, as the Protocol allows (agterm does the same).

Measured behaviour that shapes this adapter (herdr-core 0.1.18):

* ``terminal.get`` and a screen-only ``terminal.read`` (``lines: 0``) answer in
  milliseconds, so status, prompt detection and screenshots use them.
* ``terminal.read`` with history takes 1.5-10 s on an idle full-screen agent
  (it is read by scrolling). Scrollback reads are therefore time-boxed, fall
  back to the screen with ``truncated=True`` on timeout, and back off per
  terminal so a slow read is not retried on every poll.
* ``terminal.send`` pastes text and presses Enter once nobody has typed for
  about 3 s; it cannot type without submitting. ``terminal.keys`` presses
  named keys and accepts single characters only for letters, digits and
  unshifted US punctuation, so literal text without Enter is typed key by key
  and refused when it holds anything else.
* There is no ANSI read: ``with_ansi`` is ignored and screenshots render
  without colour, as on agterm.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from pathlib import Path

import structlog

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
from .hc_events import (
    is_resync,
    is_subscribed,
    open_event_stream,
    parse_status,
    snapshot_events,
    translate_event,
)
from .hc_socket import HcRpcError, HcTransport, default_socket_path, request

__all__ = [
    "HC_SUPPORTED_GENERATION",
    "HC_WINDOW_ID_PREFIX",
    "HcError",
    "HcManager",
    "address_for_window_id",
    "translate_key",
    "window_id_for_address",
]

logger = structlog.get_logger()

# Protocol generation this adapter was written against (``hello.generation``).
HC_SUPPORTED_GENERATION = 1

HC_WINDOW_ID_PREFIX = "hc-"

# ``t5`` / ``t1y`` on this core, ``machine/t7`` on a connected machine. The
# machine part is restricted to a conservative charset so a window id can never
# smuggle a delimiter, whitespace or path traversal into a request.
_ADDRESS_RE = re.compile(
    r"(?:(?P<machine>[A-Za-z0-9][A-Za-z0-9._-]{0,63})/)?t[0-9a-z]+"
)

# Label herdr-core terminals carry when ccgram's own probes started them.
_PROBE_LABEL = "ccgram.probe"
# Labels the Herdr app sets on the terminals it shows.
_WHERE_LABEL = "where"
_TAB_LABEL = "herdr.tab"
_WHERE_SEPARATOR = " / "

_HC_CAPABILITIES = MultiplexerCapabilities(
    name="hc",
    ids_stable_across_restart=True,
    exposes_pane_tty=False,
    native_agent_status=True,
    # herdr-core documents no cap; 1000 is the history size measured in step 1
    # and keeps the (slow) history reads bounded, matching the herdr backend.
    read_max_lines=1000,
    self_identify_env="HC_TERMINAL_ID",
    supports_event_stream=True,
    native_worktrees=False,
    supports_workspace_selection=True,
    native_topic_targets=False,
    supports_shell_prompt_markers=False,
)

# Request time budgets (seconds).
_CALL_TIMEOUT = 5.0
# terminal.send waits for a quiet keyboard (~3 s after the last keystroke at the
# desk) before pasting, so it needs a longer budget than a read.
_SEND_TIMEOUT = 20.0
# History reads can take 10 s; past this the screen is returned instead.
_HISTORY_READ_TIMEOUT = 2.5
# After a history read times out, serve the screen for this long before trying
# history again for that terminal, so a slow read is not repeated every poll.
_HISTORY_BACKOFF_SECONDS = 30.0

# Event stream: nothing for this long (heartbeats come every 15 s) means the
# connection is dead; reconnect with exponential backoff.
_STREAM_IDLE_TIMEOUT = 45.0
_STREAM_BACKOFF_BASE = 1.0
_STREAM_BACKOFF_MAX = 30.0

# Creating a tab: how long to look for the terminal ``tab.new`` started.
_CREATE_DISCOVERY_TIMEOUT = 10.0
_CREATE_POLL_INTERVAL = 0.25
_APP_TIMEOUT = 15.0
_RC_TIMEOUT = 124
_RC_NO_BINARY = 127

# Clearing a leftover draft before a message goes in (see ``_clear_input``).
# Claude Code draws its input box as a ``❯`` row between two horizontal rules;
# a rule may carry the session name, so it only has to be mostly rule glyphs.
_RULE_GLYPH = "─"
_RULE_MIN_GLYPHS = 20
_PROMPT_GLYPH = "❯"
# One round empties one line of the box and joins it with its neighbour: end of
# line, kill to start, kill to end (joins the next line), backspace (joins the
# previous one), forward delete. Each key is a no-op on an empty box, unlike
# Ctrl+C, which arms Claude Code's exit on an empty box.
_CLEAR_ROUND = ("ctrl+e", "ctrl+u", "ctrl+k", "backspace", "delete")
_CLEAR_MAX_ROUNDS = 40

# Login-shell wrapper the Herdr app itself uses for an agent tab, so the agent
# resolves through the user's PATH and the terminal falls back to a shell when
# it exits. Used for the ``terminal.run`` fallback when the app is unreachable.
_LOGIN_WRAPPER = '"$@"; exec "$0" -li'

# Runner contract for ``herdr-app``: ``(args) -> (rc, stdout, stderr)``.
HcAppRunner = Callable[[Sequence[str]], Awaitable[tuple[int, str, str]]]
# Event-stream opener contract: ``() -> async iterator of notification dicts``.
HcStreamOpener = Callable[[], AsyncGenerator[dict, None]]


class HcError(RuntimeError):
    """herdr-core could not be reached or answered something unusable."""


def _is_rule(line: str) -> bool:
    stripped = line.strip()
    return stripped.count(_RULE_GLYPH) >= _RULE_MIN_GLYPHS and _RULE_GLYPH in (
        stripped[0],
        stripped[-1],
    )


def input_box_rows(screen: str) -> int | None:
    """Rows of the Claude Code input box on ``screen``, or None when there is none.

    The box is the lowest pair of adjacent rules whose first row starts with
    ``❯``. A permission or question dialog has its own text under the rule, so
    it does not count as an input box.
    """
    lines = screen.splitlines()
    rules = [i for i, line in enumerate(lines) if _is_rule(line)]
    for top, bottom in reversed(list(zip(rules, rules[1:], strict=False))):
        body = lines[top + 1 : bottom]
        if body and body[0].lstrip().startswith(_PROMPT_GLYPH):
            return len(body)
    return None


# ── Identity ───────────────────────────────────────────────────────────


def window_id_for_address(address: str) -> str:
    """Return the opaque ccgram window id for a herdr-core address."""
    return f"{HC_WINDOW_ID_PREFIX}{address}"


def address_for_window_id(window_id: str) -> str | None:
    """Return the herdr-core address behind an ``hc-`` window id, or None.

    Ids round-trip through callers that case-fold them, so the prefix and the
    local ``t…`` part are matched case-insensitively and the local part is
    normalised to herdr-core's lowercase form. Anything that is not exactly an
    address is refused rather than passed to herdr-core, which would also
    resolve terminal *names*.
    """
    if window_id[: len(HC_WINDOW_ID_PREFIX)].casefold() != HC_WINDOW_ID_PREFIX:
        return None
    raw = window_id[len(HC_WINDOW_ID_PREFIX) :]
    machine, sep, local = raw.rpartition("/")
    candidate = f"{machine}{sep}{local.casefold()}"
    return candidate if _ADDRESS_RE.fullmatch(candidate) else None


def _is_remote(address: str) -> bool:
    return "/" in address


# ── Key translation ────────────────────────────────────────────────────

# tmux key names callers send (``literal=False``) → herdr-core key names.
_NAMED_KEYS: Mapping[str, str] = {
    "enter": "enter",
    "kpenter": "enter",
    "escape": "esc",
    "esc": "esc",
    "tab": "tab",
    "btab": "shift+tab",
    "space": "space",
    "bspace": "backspace",
    "backspace": "backspace",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "pageup": "pageup",
    "pgup": "pageup",
    "ppage": "pageup",
    "pagedown": "pagedown",
    "pgdn": "pagedown",
    "npage": "pagedown",
    "delete": "delete",
    "dc": "delete",
    "insert": "insert",
    "ic": "insert",
    **{f"f{n}": f"f{n}" for n in range(1, 13)},
}

# tmux modifier prefixes → herdr-core modifier words.
_MODIFIERS: Mapping[str, str] = {"C-": "ctrl", "M-": "alt", "S-": "shift"}

# Characters ``terminal.keys`` accepts as a key of their own (measured): ASCII
# letters, digits and the unshifted US punctuation. Shifted symbols (``?``,
# ``:``, ``!``…) and non-ASCII characters are rejected by herdr-core.
_PLAIN_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-=[];',./`\\")

# Byte sequences callers pass as *literal* text that are really keys (the
# toolbar's Mode button sends ``\x1b[Z`` for Shift-Tab).
_LITERAL_SEQUENCES: Mapping[str, str] = {
    "\x1b[Z": "shift+tab",
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1b[3~": "delete",
    "\x1b[5~": "pageup",
    "\x1b[6~": "pagedown",
    "\x1b[13;2u": "shift+enter",
    "\x1b": "esc",
    "\r": "enter",
    "\n": "enter",
    "\t": "tab",
    "\x7f": "backspace",
    "\x08": "backspace",
    " ": "space",
}
_LONGEST_SEQUENCE = max(len(seq) for seq in _LITERAL_SEQUENCES)
_CTRL_LETTER_MAX = 0x1A


def _char_key(char: str) -> str | None:
    """herdr-core key name that types *char*, or None if none does."""
    if len(char) != 1:
        return None
    if char in _PLAIN_KEY_CHARS:
        return char
    if "A" <= char <= "Z":
        return f"shift+{char.lower()}"
    return None


def translate_key(token: str) -> str | None:
    """Translate one tmux key name to a herdr-core key name, or None.

    Handles named keys (``Escape``, ``BSpace``, ``BTab``, ``PPage``), modifier
    chords in any combination (``C-c`` → ``ctrl+c``, ``M-Enter`` →
    ``alt+enter``, ``M-Up`` → ``alt+up``, ``S-Tab`` → ``shift+tab``) and a
    single typeable character. None for anything else, so an unknown name is
    refused instead of being typed as text.
    """
    modifiers: list[str] = []
    rest = token
    while len(rest) > len("C-") and rest[:2] in _MODIFIERS:
        modifiers.append(_MODIFIERS[rest[:2]])
        rest = rest[2:]
    named = _NAMED_KEYS.get(rest.casefold())
    if named is None:
        if len(rest) != 1:
            return None
        if modifiers:
            # A chord names the key, not the character: ``C-A`` is ``ctrl+a``.
            lowered = rest.lower()
            named = lowered if lowered in _PLAIN_KEY_CHARS else None
        else:
            named = _char_key(rest)
        if named is None:
            return None
    if not modifiers:
        return named
    if named.startswith("shift+") and "shift" in modifiers:
        named = named.removeprefix("shift+")
    return "+".join([*modifiers, named])


def _literal_to_keys(text: str) -> list[str] | None:
    """Spell literal text as herdr-core key presses, or None if impossible.

    Used for literal text that must not be submitted (``enter=False``), which
    ``terminal.send`` cannot do. Control characters become their ``ctrl+``
    chord and known escape sequences their key.
    """
    keys: list[str] = []
    index = 0
    while index < len(text):
        for size in range(min(_LONGEST_SEQUENCE, len(text) - index), 0, -1):
            sequence = _LITERAL_SEQUENCES.get(text[index : index + size])
            if sequence is not None:
                keys.append(sequence)
                index += size
                break
        else:
            char = text[index]
            code = ord(char)
            if 1 <= code <= _CTRL_LETTER_MAX:
                keys.append(f"ctrl+{chr(code + 0x60)}")
            else:
                key = _char_key(char)
                if key is None:
                    return None
                keys.append(key)
            index += 1
    return keys


# ── Small parsing helpers (private) ────────────────────────────────────


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _labels(terminal: Mapping[str, object]) -> Mapping[str, object]:
    labels = terminal.get("labels")
    return labels if isinstance(labels, Mapping) else {}


def _argv(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [part for part in value if isinstance(part, str)]


def _command_name(argv: Sequence[str]) -> str:
    """Basename of argv[0] without a login shell's leading dash."""
    if not argv:
        return ""
    return os.path.basename(argv[0]).lstrip("-")


def _validated_work_dir(work_dir: str) -> tuple[Path | None, str]:
    """Resolve *work_dir*, or return (None, reason)."""
    path = Path(work_dir).expanduser().resolve()
    if not path.exists():
        return None, f"Directory does not exist: {work_dir}"
    if not path.is_dir():
        return None, f"Not a directory: {work_dir}"
    return path, ""


def _same_dir(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return os.path.realpath(left) == os.path.realpath(right)


def _argv_matches(terminal_argv: Sequence[str], wanted: Sequence[str]) -> bool:
    """Whether a terminal's argv runs *wanted* (directly or as a wrapper's tail)."""
    if not wanted:
        return True
    return list(terminal_argv[-len(wanted) :]) == list(wanted)


def _address_in(value: object) -> str | None:
    """Find a terminal address in ``herdr-app do`` output, if it reports one."""
    if isinstance(value, Mapping):
        for key in ("terminal", "address"):
            candidate = value.get(key)
            if isinstance(candidate, str) and _ADDRESS_RE.fullmatch(candidate):
                return candidate
        for nested in value.values():
            found = _address_in(nested)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _address_in(item)
            if found:
                return found
    return None


class HcManager:
    """herdr-core backend satisfying the ``Multiplexer`` Protocol.

    Methods return ``None`` / ``[]`` / ``False`` on failure, like the other
    backends, so callers gate on the result and never on an hc error type.
    """

    @property
    def capabilities(self) -> MultiplexerCapabilities:
        """Return the static capability declaration for the hc backend."""
        return _HC_CAPABILITIES

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        transport: HcTransport | None = None,
        stream_opener: HcStreamOpener | None = None,
        app_runner: HcAppRunner | None = None,
        app_binary: str = "herdr-app",
        own_terminal_id: str | None = None,
        claude_sessions_dir: str | None = None,
    ) -> None:
        """Build the backend without touching the socket or the filesystem.

        Args:
            socket_path: herdr-core socket; defaults to ``$HC_TERMINAL_SOCKET``
                or herdr-core's default path.
            transport: ``(method, params, timeout) -> result`` override for tests.
            stream_opener: event-stream opener override for tests.
            app_runner: ``herdr-app`` runner override for tests.
            app_binary: the Herdr app's command-line executable.
            own_terminal_id: the terminal ccgram itself runs in, never offered
                or adopted; defaults to ``$HC_TERMINAL_ID``. ``""`` excludes none.
            claude_sessions_dir: Claude Code's per-process session files, read
                for session names; defaults to ``$CLAUDE_CONFIG_DIR/sessions``
                (``~/.claude/sessions``).
        """
        self._socket_path = socket_path or default_socket_path()
        self._transport: HcTransport = transport or self._socket_transport
        self._open_stream: HcStreamOpener = stream_opener or self._default_stream
        self._app_run: HcAppRunner = app_runner or self._app_subprocess_run
        self._app_binary = app_binary
        own = (
            own_terminal_id
            if own_terminal_id is not None
            else os.environ.get("HC_TERMINAL_ID", "")
        )
        self._own_address = own.strip()
        self._claude_sessions_dir = claude_sessions_dir
        # pid -> (session file mtime, session name), so a /rename is picked up.
        self._session_names: dict[int, tuple[float, str]] = {}
        # Per-address history-read backoff deadlines and in-flight reads.
        self._history_backoff: dict[str, float] = {}
        self._history_inflight: set[str] = set()

    # ── Transport plumbing (private) ───────────────────────────────────

    async def _socket_transport(
        self, method: str, params: Mapping[str, object], timeout: float
    ) -> dict:
        return await request(self._socket_path, method, params, timeout)

    async def _default_stream(self) -> AsyncGenerator[dict, None]:
        async for obj in open_event_stream(
            self._socket_path, idle_timeout=_STREAM_IDLE_TIMEOUT
        ):
            yield obj

    async def _call(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        timeout: float = _CALL_TIMEOUT,
    ) -> dict:
        """One request; raises ``HcRpcError``."""
        return await self._transport(method, params, timeout)

    async def _try(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        timeout: float = _CALL_TIMEOUT,
    ) -> dict | None:
        """One request; None on any failure (logged at debug)."""
        try:
            return await self._call(method, params, timeout=timeout)
        except HcRpcError as exc:
            logger.debug(
                "hc call failed", method=method, kind=exc.kind, error=exc.message
            )
            return None

    async def _terminal(self, window_id: str) -> dict | None:
        """The live terminal behind *window_id*, or None (gone or unreachable)."""
        address = address_for_window_id(window_id)
        if address is None:
            return None
        terminal = await self._try("terminal.get", {"terminal": address})
        if terminal is None or terminal.get("state") != "live":
            return None
        return terminal

    async def _list_terminals(self) -> list[dict] | None:
        """Every live terminal, or None when the listing failed."""
        result = await self._try("terminal.list", {})
        if result is None:
            return None
        terminals = result.get("terminals")
        if not isinstance(terminals, list):
            return None
        return [
            terminal
            for terminal in terminals
            if isinstance(terminal, dict)
            and isinstance(terminal.get("address"), str)
            and terminal.get("state") == "live"
        ]

    async def _app_subprocess_run(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Default ``herdr-app`` runner, time-boxed."""
        # Resolve to an absolute path (fork-free posix_spawn, as the other
        # backends do), lazily so construction stays free of filesystem access.
        binary = shutil.which(self._app_binary) or self._app_binary
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [binary, *args],
                capture_output=True,
                text=True,
                timeout=_APP_TIMEOUT,
                check=False,
                close_fds=False,
            )
        except subprocess.TimeoutExpired:
            return (_RC_TIMEOUT, "", "herdr-app call timed out")
        except OSError as exc:
            return (_RC_NO_BINARY, "", str(exc))
        return (completed.returncode, completed.stdout, completed.stderr)

    async def _app_json(self, args: Sequence[str]) -> object | None:
        """Run ``herdr-app <args>`` and parse its JSON output; None on failure."""
        rc, out, err = await self._app_run(args)
        if rc != 0:
            logger.debug(
                "herdr-app call failed", args=list(args), rc=rc, err=err.strip()
            )
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError, ValueError:
            logger.debug("herdr-app returned non-JSON", args=list(args))
            return None

    # ── Projection (private) ───────────────────────────────────────────

    def _is_own(self, address: str) -> bool:
        return bool(self._own_address) and address == self._own_address

    @staticmethod
    def _is_probe(terminal: Mapping[str, object]) -> bool:
        return _PROBE_LABEL in _labels(terminal)

    def _is_visible(self, terminal: Mapping[str, object]) -> bool:
        """Whether a picker may offer this terminal (not ours, not a probe, local)."""
        address = _str(terminal.get("address"))
        return not (
            self._is_own(address) or self._is_probe(terminal) or _is_remote(address)
        )

    @staticmethod
    def _provider(terminal: Mapping[str, object]) -> str:
        """The agent name a terminal reports, else one classified from its argv."""
        status = parse_status(terminal.get("status"))
        if status is not None and status.agent:
            return status.agent
        foreground = terminal.get("foreground")
        argv = _argv(foreground.get("argv")) if isinstance(foreground, Mapping) else []
        if not argv:
            return ""
        # Lazy: providers reach back into the multiplexer seam, so a
        # module-level import here would close an import cycle.
        from ..providers.process_detection import classify_provider_from_argv

        provider = classify_provider_from_argv(argv)
        return provider if provider and provider != "shell" else ""

    @staticmethod
    def _space_tab(terminal: Mapping[str, object]) -> tuple[str, str]:
        """The Herdr space and tab a terminal sits in, from its ``where`` label."""
        where = _str(_labels(terminal).get(_WHERE_LABEL))
        if where:
            space, _sep, tab = where.partition(_WHERE_SEPARATOR)
            return space.strip(), tab.strip()
        address = _str(terminal.get("address"))
        return "Herdr", _str(terminal.get("title")) or _str(
            terminal.get("name")
        ) or address

    def _sessions_dir(self) -> Path:
        if self._claude_sessions_dir is not None:
            return Path(self._claude_sessions_dir)
        base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
        return Path(base) / "sessions"

    def _claude_session_name(self, terminal: Mapping[str, object]) -> str:
        """The Claude Code session name (``/rename``) of the terminal's agent.

        Claude Code writes ``sessions/<pid>.json`` with a ``name`` for each running
        process. Empty when the foreground isn't a named Claude session.
        """
        foreground = terminal.get("foreground")
        pid = foreground.get("pid") if isinstance(foreground, Mapping) else None
        if not isinstance(pid, int) or pid <= 0:
            return ""
        path = self._sessions_dir() / f"{pid}.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._session_names.pop(pid, None)
            return ""
        cached = self._session_names.get(pid)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return ""
        name = raw.get("name") if isinstance(raw, dict) else None
        name = name.strip() if isinstance(name, str) else ""
        self._session_names[pid] = (mtime, name)
        return name

    def _base_name(self, terminal: Mapping[str, object]) -> str:
        """A terminal's topic name before disambiguation: session name, else tab."""
        return self._claude_session_name(terminal) or self._space_tab(terminal)[1]

    def _shared_tabs(self, terminals: Sequence[Mapping[str, object]]) -> frozenset[str]:
        """Tab names that more than one visible terminal uses."""
        seen: dict[str, int] = {}
        for terminal in terminals:
            if self._is_visible(terminal):
                name = self._base_name(terminal)
                seen[name] = seen.get(name, 0) + 1
        return frozenset(tab for tab, count in seen.items() if count > 1)

    def _to_window(
        self,
        terminal: Mapping[str, object],
        shared_tabs: frozenset[str] = frozenset(),
    ) -> WindowRef:
        """Map one herdr-core terminal onto the neutral ``WindowRef``.

        Topic headers are narrow, so the name is the Claude Code session name
        (``dig-cua``), else the Herdr tab (``chore``), with the space added only
        when another terminal shares the name (``research (Tabula)``). The
        provider prefix is added by the topic layer.
        """
        address = _str(terminal.get("address"))
        status = parse_status(terminal.get("status"))
        provider = self._provider(terminal)
        foreground = terminal.get("foreground")
        fg_argv = (
            _argv(foreground.get("argv")) if isinstance(foreground, Mapping) else []
        )
        space, _tab = self._space_tab(terminal)
        base = self._base_name(terminal)
        name = f"{base} ({space})" if base in shared_tabs else base
        cols, rows = terminal.get("cols"), terminal.get("rows")
        return WindowRef(
            window_id=window_id_for_address(address),
            window_name=name,
            cwd=_str(terminal.get("cwd")),
            pane_current_command=provider or _command_name(fg_argv),
            pane_tty="",
            pane_width=cols if isinstance(cols, int) else 0,
            pane_height=rows if isinstance(rows, int) else 0,
            # Adopt only terminals whose program reports an agent status, and
            # never ccgram's own terminal, a probe, or a remote terminal whose
            # transcripts ccgram cannot read.
            topic_eligible=status is not None and self._is_visible(terminal),
        )

    # ── Multiplexer Protocol surface ───────────────────────────────────

    async def ensure_session(self) -> None:
        """Verify herdr-core answers ``hello``; warn on an unverified generation.

        Raises:
            HcError: the socket is unreachable or the handshake is unusable.
        """
        try:
            hello = await self._call("hello", {})
        except HcRpcError as exc:
            raise HcError(
                f"herdr-core is not reachable at {self._socket_path}: {exc.message}"
            ) from exc
        generation = hello.get("generation")
        if generation != HC_SUPPORTED_GENERATION:
            logger.warning(
                "herdr-core protocol generation is unverified; continuing",
                generation=generation,
                supported_generation=HC_SUPPORTED_GENERATION,
                version=hello.get("version"),
            )

    async def list_windows(self) -> list[WindowRef]:
        """Terminals a picker may offer ([] on failure).

        Visibility only: ccgram's own terminal, probe terminals and remote
        terminals are hidden. A local shell without agent status is listed
        with ``topic_eligible=False`` so a user can still bind it by hand.
        """
        terminals = await self._list_terminals()
        if terminals is None:
            return []
        shared = self._shared_tabs(terminals)
        return [self._to_window(t, shared) for t in terminals if self._is_visible(t)]

    async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
        """Every live terminal, or None when herdr-core could not answer.

        Complete on purpose: hidden terminals stay in, stamped ineligible, so
        cleanup never mistakes a filtered-out terminal for a departed one.
        """
        terminals = await self._list_terminals()
        if terminals is None:
            return None
        shared = self._shared_tabs(terminals)
        return [self._to_window(t, shared) for t in terminals]

    async def window_exists(self, window_id: str) -> bool | None:
        """Ask herdr-core about one terminal: True, False, or None if unknown.

        Authoritative and cheap (``terminal.get`` answers in well under a
        millisecond). ``not_found`` and a finished terminal are proof of
        absence; anything else (an unreachable socket or machine) is unknown.
        """
        address = address_for_window_id(window_id)
        if address is None:
            return None
        try:
            terminal = await self._call("terminal.get", {"terminal": address})
        except HcRpcError as exc:
            return False if exc.kind == "not_found" else None
        state = terminal.get("state")
        if state == "live":
            return True
        if state == "finished":
            return False
        return None

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """Herdr spaces on this computer (``herdr-app ls``); [] on failure."""
        spaces = await self._spaces()
        return [
            WorkspaceRef(
                workspace_id=_str(space.get("id")),
                label=_str(space.get("name")),
                cwd=_str(space.get("folder")),
            )
            for space in spaces
        ]

    async def _spaces(self) -> list[Mapping[str, object]]:
        """Local spaces from ``herdr-app ls``; [] when the app is unreachable."""
        listing = await self._app_json(["ls"])
        raw = listing.get("spaces") if isinstance(listing, Mapping) else None
        if not isinstance(raw, list):
            return []
        return [
            space
            for space in raw
            if isinstance(space, Mapping)
            and _str(space.get("id"))
            # A space on another machine would start a remote terminal.
            and not space.get("machine")
        ]

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """The live terminal behind *window_id*, or None."""
        terminal = await self._terminal(window_id)
        if terminal is None:
            return None
        # Name it the way the listing does, so the topic title doesn't flip.
        terminals = await self._list_terminals()
        return self._to_window(terminal, self._shared_tabs(terminals or []))

    async def _read(self, window_id: str, lines: int) -> dict | None:
        address = address_for_window_id(window_id)
        if address is None:
            return None
        timeout = _CALL_TIMEOUT if lines == 0 else _HISTORY_READ_TIMEOUT
        return await self._call(
            "terminal.read", {"terminal": address, "lines": lines}, timeout=timeout
        )

    async def _read_screen(self, window_id: str) -> str | None:
        """Screen-only text (fast); None on failure."""
        try:
            result = await self._read(window_id, 0)
        except HcRpcError as exc:
            logger.debug("hc screen read failed", window_id=window_id, kind=exc.kind)
            return None
        text = result.get("text") if result else None
        return text if isinstance(text, str) else None

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Screen text, trailing whitespace stripped; None on failure or empty.

        ``with_ansi`` is accepted and ignored: herdr-core reads plain text.
        """
        del with_ansi
        text = await self._read_screen(window_id)
        return (text.rstrip() or None) if text is not None else None

    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,
    ) -> str | None:
        """Reject raw pane locators; only the window id authorises a read."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw hc pane read")
            return None
        return await self.capture_pane(window_id, with_ansi=with_ansi)

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Screen plus up to ``read_max_lines`` of history, time-boxed.

        A history read that exceeds its budget (idle full-screen agents take
        seconds) yields the screen with ``truncated=True`` and backs off that
        terminal for a while, so the poll loop never waits on it twice.
        """
        address = address_for_window_id(window_id)
        if address is None:
            return None
        cap = self.capabilities.read_max_lines or lines
        effective = max(0, min(lines, cap))
        truncated = effective != lines
        if effective == 0:
            text = await self._read_screen(window_id)
            return None if text is None else CaptureResult(text, truncated)
        backing_off = self._history_backoff.get(address, 0.0) > time.monotonic()
        if backing_off or address in self._history_inflight:
            text = await self._read_screen(window_id)
            return None if text is None else CaptureResult(text, truncated=True)
        self._history_inflight.add(address)
        try:
            result = await self._read(window_id, effective)
        except HcRpcError as exc:
            if exc.kind != "timeout":
                logger.debug(
                    "hc history read failed", window_id=window_id, kind=exc.kind
                )
                return None
            logger.info(
                "hc history read timed out; serving the screen instead",
                window_id=window_id,
                timeout=_HISTORY_READ_TIMEOUT,
            )
            self._history_backoff[address] = time.monotonic() + _HISTORY_BACKOFF_SECONDS
            text = await self._read_screen(window_id)
            return None if text is None else CaptureResult(text, truncated=True)
        finally:
            self._history_inflight.discard(address)
        self._history_backoff.pop(address, None)
        if not result:
            return None
        text = result.get("text")
        if not isinstance(text, str):
            return None
        # ``complete: false`` means herdr-core may have left part of it out.
        return CaptureResult(text, truncated or result.get("complete") is False)

    async def capture_pane_scrollback(
        self, window_id: str, history: int = 200
    ) -> str | None:
        """Scrollback text as a plain string (legacy alias)."""
        result = await self.capture_scrollback(window_id, lines=history)
        return result.text if result else None

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Columns and rows from ``terminal.get``."""
        terminal = await self._terminal(window_id)
        if terminal is None:
            return None
        cols, rows = terminal.get("cols"), terminal.get("rows")
        if not isinstance(cols, int) or not isinstance(rows, int):
            return None
        return PaneDims(width=cols, height=rows)

    async def _keys(self, address: str, keys: Sequence[str]) -> bool:
        try:
            await self._call(
                "terminal.keys",
                {"terminal": address, "keys": list(keys)},
                timeout=_SEND_TIMEOUT,
            )
        except HcRpcError as exc:
            logger.debug("hc keys failed", terminal=address, kind=exc.kind, keys=keys)
            return False
        return True

    async def _clear_input(self, window_id: str, address: str) -> None:
        """Empty the agent's input box so a pasted message is not appended to a draft.

        ``terminal.send`` pastes at the cursor, so text left in the box at the
        desk (a half-typed ``/rename ``) would prefix the message and change what
        it means. The message from Telegram wins over the draft. The plain-text
        screen cannot tell a draft from Claude Code's dimmed placeholder, so the
        box is always cleared, with one round of no-op-when-empty keys per row.
        Best effort: an unreadable screen or an unrecognised UI sends as before.
        """
        screen = await self._read_screen(window_id)
        rows = input_box_rows(screen) if screen else None
        if rows is None:
            return
        rounds = min(rows + 1, _CLEAR_MAX_ROUNDS)
        await self._keys(address, list(_CLEAR_ROUND) * rounds)

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,
    ) -> bool:
        """Send text or keys to the terminal.

        * Literal text with Enter first clears the input box (``_clear_input``),
          then goes to ``terminal.send``: a bracketed paste
          followed by Enter once the keyboard has been quiet for ~3 s, so a
          person typing at the desk is never interleaved mid-line. ``force``
          is set because ccgram already validates prompts before sending,
          matching the other backends, which type regardless.
        * Literal text without Enter is spelled out with ``terminal.keys``
          (``terminal.send`` always submits). Text holding characters no key
          types (shifted symbols, non-ASCII) is refused.
        * ``literal=False``: whitespace-separated tmux key names, translated by
          ``translate_key``; an unknown name refuses the whole send.

        ``raw`` is accepted for contract compatibility; herdr-core needs none
        of the tmux TUI workarounds it disables.
        """
        del raw
        address = address_for_window_id(window_id)
        if address is None:
            return False
        if not literal:
            keys: list[str] = []
            for token in text.split():
                key = translate_key(token)
                if key is None:
                    logger.warning("Unknown key name for hc", token=token)
                    return False
                keys.append(key)
            if enter:
                keys.append("enter")
            return bool(keys) and await self._keys(address, keys)
        if enter and text:
            await self._clear_input(window_id, address)
            try:
                await self._call(
                    "terminal.send",
                    {"terminal": address, "text": text, "force": True},
                    timeout=_SEND_TIMEOUT,
                )
            except HcRpcError as exc:
                logger.debug("hc send failed", terminal=address, kind=exc.kind)
                return False
            return True
        if not text:
            return enter and await self._keys(address, ["enter"])
        typed = _literal_to_keys(text)
        if typed is None:
            logger.warning(
                "hc cannot type this text without submitting it",
                window_id=window_id,
                length=len(text),
            )
            return False
        return await self._keys(address, typed)

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

    async def send_to_pane(
        self,
        pane_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        window_id: str | None = None,
        raw: bool = True,
    ) -> bool:
        """Reject raw pane locators; only the window id authorises a write."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw hc pane write")
            return False
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

    async def kill_window(self, window_id: str) -> bool:
        """End the terminal's program (``terminal.kill``)."""
        address = address_for_window_id(window_id)
        if address is None:
            return False
        return await self._try("terminal.kill", {"terminal": address}) is not None

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename the Herdr tab the terminal sits in (``herdr-app do tab.rename``)."""
        terminal = await self._terminal(window_id)
        if terminal is None:
            return False
        tab = _str(_labels(terminal).get(_TAB_LABEL))
        if not tab:
            # A terminal started without the app (the ``terminal.run``
            # fallback) has no tab to rename.
            return False
        rc, _out, err = await self._app_run(
            ["do", "tab.rename", json.dumps({"id": tab, "name": new_name})]
        )
        if rc != 0:
            logger.debug("herdr-app tab.rename failed", rc=rc, err=err.strip())
        return rc == 0

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """No pane handles: one terminal is one window with one pane."""
        del window_id
        return []

    async def split_window(self, window_id: str) -> str | None:
        """Always None: herdr-core panes are app layout, not addressable here."""
        del window_id
        return None

    # ── Creation ───────────────────────────────────────────────────────

    async def _space_for(self, path: Path) -> str | None:
        """The local space whose folder holds *path* most specifically."""
        best: tuple[int, str] | None = None
        for space in await self._spaces():
            folder = _str(space.get("folder"))
            if not folder:
                continue
            root = Path(os.path.realpath(Path(folder).expanduser()))
            if path == root or root in path.parents:
                depth = len(root.parts)
                if best is None or depth > best[0]:
                    best = (depth, _str(space.get("id")))
        return best[1] if best else None

    async def _find_new_terminal(
        self,
        before: set[str],
        path: Path,
        argv: Sequence[str],
        *,
        timeout: float,
    ) -> str | None:
        """Address of the one new local terminal running *argv* in *path*.

        Polls ``terminal.list`` until exactly one terminal that was not in
        *before* matches the folder and argv, or *timeout* passes. Several
        matches are ambiguous and return None rather than guess.
        """
        deadline = time.monotonic() + timeout
        while True:
            terminals = await self._list_terminals()
            if terminals is not None:
                matches = [
                    _str(t.get("address"))
                    for t in terminals
                    if _str(t.get("address")) not in before
                    and not _is_remote(_str(t.get("address")))
                    and _same_dir(_str(t.get("cwd")), str(path))
                    and _argv_matches(_argv(t.get("argv")), argv)
                ]
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    logger.warning(
                        "hc cannot tell which new terminal is ours",
                        candidates=matches,
                    )
                    return None
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(_CREATE_POLL_INTERVAL)

    async def _run_without_app(
        self, path: Path, argv: Sequence[str]
    ) -> tuple[str | None, str]:
        """Start the terminal through ``terminal.run`` (no Herdr tab)."""
        shell = os.environ.get("SHELL") or "/bin/zsh"
        params: dict[str, object] = {
            "cwd": str(path),
            # Never a child of the terminal ccgram runs in: children end with
            # their parent.
            "no_parent": True,
            "no_notify": True,
            "keep": True,
        }
        if argv:
            params["argv"] = [shell, "-lic", _LOGIN_WRAPPER, shell, *argv]
        try:
            result = await self._call("terminal.run", params, timeout=_APP_TIMEOUT)
        except HcRpcError as exc:
            return None, f"herdr-core could not start the terminal: {exc.message}"
        address = _str(result.get("address"))
        if not _ADDRESS_RE.fullmatch(address):
            return None, "herdr-core returned no terminal address"
        return address, ""

    async def _open_tab(
        self,
        path: Path,
        argv: Sequence[str],
        before: set[str],
        *,
        window_name: str | None,
        workspace_id: str | None,
    ) -> tuple[str | None, str]:
        """Start the terminal in a new Herdr tab; return ``(address, error)``."""
        payload: dict[str, object] = {"folder": str(path)}
        if argv:
            payload["argv"] = list(argv)
        if window_name:
            payload["name"] = window_name
        space = workspace_id or await self._space_for(path)
        if space:
            payload["space"] = space
        rc, out, err = await self._app_run(["do", "tab.new", json.dumps(payload)])
        if rc != 0:
            logger.info(
                "herdr-app tab.new failed; starting the terminal without a tab",
                rc=rc,
                err=err.strip(),
            )
            # A timed-out app call may still have opened the tab; look once
            # before starting a second terminal.
            address = await self._find_new_terminal(before, path, argv, timeout=0.0)
            if address is not None:
                return address, ""
            return await self._run_without_app(path, argv)
        reported = None
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            reported = _address_in(json.loads(out))
        if reported and reported not in before:
            return reported, ""
        address = await self._find_new_terminal(
            before, path, argv, timeout=_CREATE_DISCOVERY_TIMEOUT
        )
        if address is None:
            return None, "Herdr opened a tab, but its terminal could not be identified"
        return address, ""

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
        """Open a Herdr tab running the agent, then identify its terminal.

        ``herdr-app do tab.new`` makes the session a normal tab in the app.
        The app does not report the terminal it started, so the new terminal is
        found by diffing ``terminal.list`` against a listing taken before,
        matching the folder and argv, within a bounded wait. When the app is
        unreachable (and nothing new appeared), the terminal is started with
        ``terminal.run`` instead; it then has no tab until the app shows it.
        """
        path, problem = _validated_work_dir(work_dir)
        if path is None:
            return (False, problem, "", "")
        command = launch_command or ""
        if command and agent_args:
            command = f"{command} {agent_args}"
        try:
            argv = shlex.split(command) if start_agent and command else []
        except ValueError as exc:
            return (False, f"Cannot parse launch command {command!r}: {exc}", "", "")

        listing = await self._list_terminals()
        if listing is None:
            return (False, "herdr-core is not reachable", "", "")
        before = {_str(t.get("address")) for t in listing}
        address, message = await self._open_tab(
            path, argv, before, window_name=window_name, workspace_id=workspace_id
        )
        if address is None:
            return (False, message, "", "")
        return (True, "", window_name or path.name, window_id_for_address(address))

    async def create_topic_target(
        self,
        work_dir: str,
        *,
        launch_command: str | None,
        workspace_id: str | None,
        window_name: str | None = None,
        agent_args: str = "",
    ) -> TopicTargetResult:
        """Create a topic target; the terminal's window id is its durable target."""
        success, message, label, window_id = await self.create_window(
            work_dir,
            window_name=window_name,
            launch_command=launch_command,
            agent_args=agent_args,
            workspace_id=workspace_id,
        )
        if not success:
            raise RuntimeError(message)
        return TopicTargetResult(target_id=window_id, label=label, window_id=window_id)

    async def create_worktree_window(
        self,
        repo_path: str,
        worktree_path: str,
        branch: str,
        *,
        window_name: str | None = None,
        launch_command: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Not supported (``native_worktrees`` is False)."""
        del repo_path, worktree_path, branch, window_name, launch_command
        return (False, "hc does not create worktrees natively", "", "")

    # ── Process and status ─────────────────────────────────────────────

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """The terminal's foreground process (pid, argv, cwd; no pgid, no tty)."""
        terminal = await self._terminal(window_id)
        if terminal is None:
            return None
        foreground = terminal.get("foreground")
        if not isinstance(foreground, Mapping):
            return None
        argv = _argv(foreground.get("argv"))
        if not argv:
            return None
        pid = foreground.get("pid")
        return ForegroundInfo(
            pid=pid if isinstance(pid, int) else 0,
            pgid=0,
            argv=argv,
            cwd=_str(foreground.get("cwd")),
            tty="",
        )

    async def agent_status(self, window_id: str) -> AgentStatus | None:
        """The status the program reports (``status.state`` / ``app`` / ``msg``)."""
        terminal = await self._terminal(window_id)
        if terminal is None:
            return None
        return parse_status(terminal.get("status"))

    async def watch_events(
        self, window_ids: Sequence[str]
    ) -> AsyncGenerator[MuxEvent, None]:
        """Stream push events for *window_ids* (see ``Multiplexer.watch_events``).

        Holds one ``events.subscribe`` connection. The subscribe snapshot
        reprimes agent status for the watched terminals; afterwards ``status``
        events become ``agent_status`` and ``finished`` / ``lost`` /
        ``removed`` become ``window_died``. Heartbeats and other types are
        ignored. A dropped or silent connection reconnects with exponential
        backoff; a ``missed`` notice re-subscribes at once for a fresh snapshot.
        """
        watched = {
            address: window_id
            for window_id in window_ids
            if (address := address_for_window_id(window_id)) is not None
        }
        backoff = _STREAM_BACKOFF_BASE
        while True:
            resync = False
            try:
                async with contextlib.aclosing(self._open_stream()) as stream:
                    async for obj in stream:
                        if is_subscribed(obj):
                            backoff = _STREAM_BACKOFF_BASE
                            terminals = obj.get("terminals")
                            for event in snapshot_events(
                                terminals if isinstance(terminals, list) else [],
                                watched,
                            ):
                                yield event
                            continue
                        if is_resync(obj):
                            resync = True
                            break
                        for event in translate_event(obj, watched):
                            yield event
            except (OSError, TimeoutError, ValueError, HcRpcError) as exc:
                logger.debug("hc event stream error: %s", exc)
            if resync:
                logger.info("hc event stream fell behind; re-subscribing")
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _STREAM_BACKOFF_MAX)

    async def get_pane_title(self, window_id: str) -> str:
        """The terminal title, or '' when it reports none."""
        terminal = await self._terminal(window_id)
        return _str(terminal.get("title")) if terminal else ""

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """No-op: herdr-core has no call to set a terminal title.

        Provider re-detection uses the reported agent status and the
        foreground argv instead.
        """
        del window_id, provider_name
