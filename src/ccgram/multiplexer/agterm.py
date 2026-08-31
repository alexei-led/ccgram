"""agterm backend for the Multiplexer contract, via the agtermctl CLI/socket.

Anti-corruption layer over `agterm <https://github.com/umputun/agterm>`_'s
unix-socket control channel, driven through its ``agtermctl`` CLI. Every agterm
JSON shape (``tree`` / ``workspaces[]`` / ``sessions[]`` / ``surfaces[]``) and
every ``surface:<uuid>:<role>`` locator stays **private** to this module;
callers see only the neutral value types from ``multiplexer.base``.

Identity: an agterm *session* is the unit that carries one agent, so its UUID is
the neutral ``window_id``. agterm persists that UUID in its per-window state file
and restores it, so unlike herdr the id survives a restart and no alias
reconciliation is needed (``ids_stable_across_restart`` is True).

The backend shells out to ``agtermctl`` (the seam explicitly allows a CLI in
place of speaking the socket directly); the socket path is passed through
``--socket``. The command runner is injectable so unit tests feed JSON fixtures
without a live socket and the constructor stays I/O-free.

Two agterm behaviours drive the shape of this adapter:

* **Read and write default to different panes.** ``session text`` with no
  ``--pane`` reads the visible or focused pane, while ``session type`` with no
  ``--pane`` injects into the main pane. Every capture and injection below
  therefore passes an explicit role and never relies on a default. The agent
  ccgram drives lives in the main pane, so that role is ``left`` throughout.
* **A split pane has no durable handle.** agterm promotes the split survivor
  into the main pane when the primary exits, so the ``right`` role is not a
  stable identity. ``split_window`` returns None and ``list_panes`` returns [],
  exactly as the Protocol prescribes for a backend that cannot expose a safe
  sibling handle.

Capabilities: ``native_agent_status`` is **False** even though agterm reports a
status natively. In this codebase that flag also selects herdr-shaped creation,
workspace and topic-eligibility behaviour (``handlers/topics/window_launch_service.py``,
``handlers/split_command.py``, ``multiplexer/topic_mapping.py``), so declaring it
True would stop agterm windows from ever surfacing as topics. Adopting agterm's
native status needs that conflation untangled upstream first; until then ccgram's
terminal scraping supplies status, as it does for tmux. ``exposes_pane_tty`` is
False (agterm reports foreground argv, no tty or pid), ``read_max_lines`` is None
(``session text --lines N`` takes any N; agterm imposes no cap), and ``supports_event_stream`` is False
(the control channel has no event subscription).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

import structlog

from ..config import config
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

__all__ = ["AgtermManager"]


class _Unset:
    """Sentinel: ``None`` is a meaningful workspace scope (every workspace)."""


_UNSET = _Unset()

logger = structlog.get_logger()

# ``(args, stdin_text) -> (rc, stdout, stderr)``; injected in tests.
AgtermRunner = Callable[[Sequence[str], str | None], Awaitable[tuple[int, str, str]]]

_CALL_TIMEOUT_SECONDS = 10
_RC_TIMEOUT = 124
_RC_NO_BINARY = 127

# The agent ccgram drives runs in the session's main pane. Passed explicitly on
# every read and write because agterm's two defaults disagree (see module docs).
_AGENT_PANE = "left"

# ``C-x``: a control chord is exactly the prefix plus one character.
_CONTROL_CHORD_LEN = 3

# The gap between typing text and submitting it, matching the tmux backend.
_SUBMIT_DELAY_SECONDS = 0.5

# How long to wait for a new session's terminal to come up before giving up.
_REALIZE_ATTEMPTS = 40
_REALIZE_POLL_SECONDS = 0.05

# Bracketed paste (DEC mode 2004). A TUI that has enabled it treats everything
# between the markers as literal text, so interior newlines land as newlines
# instead of submits. ``session type`` adds no markers of its own.
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_AGTERM_CAPABILITIES = MultiplexerCapabilities(
    name="agterm",
    ids_stable_across_restart=True,
    exposes_pane_tty=False,
    native_agent_status=False,
    read_max_lines=None,
    self_identify_env="AGTERM_SESSION_ID",
    supports_event_stream=False,
    native_worktrees=False,
)

# tmux key names (callers send these with ``literal=False``) → the bytes a pty
# expects. agterm's ``session type`` injects every byte it is given verbatim
# apart from newlines, which it turns into a Return keypress, so a key table is
# all that is needed to satisfy the contract.
_KEY_BYTES: dict[str, str] = {
    "Enter": "\r",
    "Escape": "\x1b",
    "Tab": "\t",
    "Space": " ",
    "BSpace": "\x7f",
    "BTab": "\x1b[Z",
    "Up": "\x1b[A",
    "Down": "\x1b[B",
    "Right": "\x1b[C",
    "Left": "\x1b[D",
    "Home": "\x1b[H",
    "End": "\x1b[F",
    "PageUp": "\x1b[5~",
    "PageDown": "\x1b[6~",
    "Delete": "\x1b[3~",
}

# The handful of non-letter control chords tmux spells as ``C-<symbol>``.
_CONTROL_SYMBOLS: dict[str, str] = {
    "@": "\x00",
    "[": "\x1b",
    "\\": "\x1c",
    "]": "\x1d",
    "^": "\x1e",
    "_": "\x1f",
}


def _validated_work_dir(work_dir: str) -> tuple[Path | None, str]:
    """Resolve *work_dir*, or return (None, reason).

    agterm accepts an arbitrary cwd and can report a session created before its
    surface fails, so an unusable path has to be refused here, as the tmux
    backend refuses it.
    """
    path = Path(work_dir).expanduser().resolve()
    if not path.exists():
        return None, f"Directory does not exist: {work_dir}"
    if not path.is_dir():
        return None, f"Not a directory: {work_dir}"
    return path, ""


def _key_to_bytes(token: str) -> str | None:
    """Translate one tmux key name to pty bytes, or None when unrecognised.

    Handles the three shapes callers actually send: a named key (``Escape``),
    a control chord (``C-c``), and a meta chord (``M-Enter``). Returning None
    for anything else keeps an unknown name from being typed as literal text.
    """
    if token in _KEY_BYTES:
        return _KEY_BYTES[token]
    if token.startswith("M-"):
        rest = token[2:]
        # Meta plus a single printable character is ESC then that character —
        # the shipped Think button sends ``M-t``, which no named-key table can
        # cover. Anything longer must resolve through the table.
        if len(rest) == 1 and rest.isprintable():
            return "\x1b" + rest
        resolved = _key_to_bytes(rest)
        return None if resolved is None else "\x1b" + resolved
    if token.startswith("C-") and len(token) == _CONTROL_CHORD_LEN:
        char = token[2]
        if char.isalpha():
            return chr(ord(char.upper()) - 0x40)
        return _CONTROL_SYMBOLS.get(char)
    return None


class AgtermManager:
    """Drives agterm sessions through ``agtermctl``, behind the Multiplexer contract.

    Returns the neutral value types and exposes ``capabilities``. All agterm JSON
    parsing is private; methods return ``None``/``[]``/``False`` on failure exactly
    like the tmux and herdr backends, so callers gate on the result and never on an
    agterm-specific error type.
    """

    @property
    def capabilities(self) -> MultiplexerCapabilities:
        """Return the static capability declaration for the agterm backend."""
        return _AGTERM_CAPABILITIES

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        binary: str = "agtermctl",
        runner: AgtermRunner | None = None,
        own_session_id: str | None = None,
        workspaces: tuple[str, ...] | None | _Unset = _UNSET,
    ) -> None:
        """Build the backend without touching the socket (I/O-free).

        Args:
            socket_path: agterm control socket; defaults to ``$AGTERM_SOCKET``.
            binary: the ``agtermctl`` executable name/path.
            runner: async ``(args, stdin) -> (rc, stdout, stderr)`` override for tests.
            own_session_id: the session ccgram runs in, excluded from listings;
                defaults to ``$AGTERM_SESSION_ID``. Pass ``""`` to exclude none.
            workspaces: workspace names discovery may adopt from; ``None`` means
                every workspace. Defaults to ``config.agterm_workspaces``.
        """
        self._socket_path = socket_path or os.environ.get("AGTERM_SOCKET", "")
        # Resolve to an absolute path so CPython takes the fork-free
        # ``posix_spawn`` path, as the herdr backend does and for the same
        # reason: forking from this long-lived async process makes children
        # emit macOS MallocStackLogging noise.
        self._binary = shutil.which(binary) or binary
        self._run: AgtermRunner = runner or self._subprocess_run
        # The session the bot itself runs in, so listings can skip it the way
        # the tmux backend skips ``config.own_window_id``. Empty when ccgram
        # runs outside agterm, in which case nothing is excluded on this count.
        self._own_session_id = (
            own_session_id
            if own_session_id is not None
            else os.environ.get("AGTERM_SESSION_ID", "")
        ).casefold()
        scope = (
            config.agterm_workspaces if isinstance(workspaces, _Unset) else workspaces
        )
        self._workspaces = (
            None if scope is None else {name.casefold() for name in scope}
        )

    # ── CLI plumbing (private) ─────────────────────────────────────────

    async def _subprocess_run(
        self, args: Sequence[str], stdin_text: str | None = None
    ) -> tuple[int, str, str]:
        """Default runner: exec ``agtermctl <args>``, time-boxed."""
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [self._binary, *args],
                capture_output=True,
                text=True,
                input=stdin_text,
                timeout=_CALL_TIMEOUT_SECONDS,
                check=False,
                close_fds=False,
            )
        except subprocess.TimeoutExpired:
            return (_RC_TIMEOUT, "", "agtermctl call timed out")
        except OSError as exc:
            return (_RC_NO_BINARY, "", str(exc))
        return (completed.returncode, completed.stdout, completed.stderr)

    def _with_socket(self, args: Sequence[str]) -> list[str]:
        """Append ``--json`` and, when known, the explicit socket path."""
        out = [*args, "--json"]
        if self._socket_path:
            out += ["--socket", self._socket_path]
        return out

    async def _call(
        self, args: Sequence[str], stdin_text: str | None = None
    ) -> dict | None:
        """Run ``agtermctl <args> --json`` and return the ``result`` dict, or None.

        None on: non-zero exit (agterm not running, unknown target), non-JSON
        output, or an ``ok: false`` envelope. Failures log at debug — callers
        treat None as "session gone / call failed", matching the other backends.
        """
        rc, out, err = await self._run(self._with_socket(args), stdin_text)
        if rc != 0:
            logger.debug("agterm call failed", args=list(args), rc=rc, err=err.strip())
            return None
        try:
            payload = json.loads(out)
        except json.JSONDecodeError, ValueError:
            logger.debug("agterm returned non-JSON", args=list(args))
            return None
        if not isinstance(payload, dict) or not payload.get("ok"):
            logger.debug("agterm error payload", args=list(args), payload=payload)
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    async def _call_ok(
        self, args: Sequence[str], stdin_text: str | None = None
    ) -> bool:
        """Run a mutating command; True when agterm reported ``ok``."""
        return await self._call(args, stdin_text) is not None

    async def _open_window_ids(self) -> list[str] | None:
        """Ids of agterm's open windows, or None when the query fails.

        Closed windows are filtered out: agterm keeps them in the library and
        answers ``window not open`` for a tree read, which under the
        fail-closed rule below would make every listing unavailable forever.
        """
        result = await self._call(["window", "list"])
        if result is None:
            return None
        windows = result.get("windows")
        if not isinstance(windows, list):
            return None
        return [
            str(window["id"])
            for window in windows
            if isinstance(window, dict) and window.get("id") and window.get("open")
        ]

    async def _tree(self) -> dict | None:
        """Return a tree merged across every open window, or None on failure.

        ``tree`` is placement-scoped: given no ``--window`` it answers for the
        frontmost window alone (``controlTree`` resolves a nil window through
        ``resolvePlacementStore`` to ``library.activeStore``). Trusting that
        would hide every session in any other window, and since the listing
        looks complete, reconciliation would read those absences as death and
        prune live state. Session commands take ``--target`` and do resolve
        across windows, so this read is the only one that has to be widened.

        Fails closed. A failure anywhere returns None, never a partial tree,
        because a partial answer here is indistinguishable from sessions
        having gone away.
        """
        window_ids = await self._open_window_ids()
        if window_ids is None:
            return None
        workspaces: list[dict] = []
        for window_id in window_ids:
            result = await self._call(["tree", "--window", window_id])
            if result is None:
                return None
            tree = result.get("tree")
            if not isinstance(tree, dict):
                return None
            workspaces.extend(
                workspace
                for workspace in tree.get("workspaces", [])
                if isinstance(workspace, dict)
            )
        return {"workspaces": workspaces}

    @staticmethod
    def _sessions(tree: dict) -> list[tuple[dict, dict]]:
        """Flatten the tree to ``(workspace, session)`` pairs."""
        pairs: list[tuple[dict, dict]] = []
        for workspace in tree.get("workspaces", []):
            if not isinstance(workspace, dict):
                continue
            for session in workspace.get("sessions", []):
                if isinstance(session, dict) and session.get("id"):
                    pairs.append((workspace, session))
        return pairs

    @staticmethod
    def _to_window(session: dict) -> WindowRef:
        """Map one agterm session node onto the neutral ``WindowRef``."""
        foreground = session.get("foreground")
        command = ""
        if isinstance(foreground, list) and foreground:
            command = str(foreground[0])
        return WindowRef(
            window_id=str(session.get("id", "")),
            window_name=str(session.get("name") or ""),
            cwd=str(session.get("cwd") or ""),
            pane_current_command=command,
            # agterm exposes no tty and no pane geometry over the control
            # channel; the capability flags declare both absences.
            pane_tty="",
            pane_width=0,
            pane_height=0,
        )

    async def _find_session(self, window_id: str) -> dict | None:
        """Return the private session dict for *window_id*, or None if gone.

        agterm ids are UUID strings whose case is not guaranteed to match what a
        caller persisted (the environment exports them uppercase, the tree
        reports them uppercase, but a caller may round-trip them lowercased), so
        the comparison is case-insensitive.
        """
        tree = await self._tree()
        if tree is None:
            return None
        wanted = window_id.casefold()
        for _workspace, session in self._sessions(tree):
            if str(session.get("id", "")).casefold() == wanted:
                return session
        return None

    def _in_scope(self, workspace: dict) -> bool:
        """Whether discovery may adopt from this workspace.

        agterm has no per-application container the way tmux has its own
        session, so without a scope every session the user has open surfaces as
        a topic. One named workspace is the analogue of ``TMUX_SESSION_NAME``.
        """
        if self._workspaces is None:
            return True
        return str(workspace.get("name") or "").casefold() in self._workspaces

    def _is_adoptable(self, session: dict) -> bool:
        """Whether discovery may bind a Telegram topic to this session.

        Mirrors the exclusions the other backends apply, because every window a
        listing returns is a window discovery can auto-adopt:

        * the session ccgram itself runs in, matching tmux's ``own_window_id``
          skip. On agterm that is simply the ``AGTERM_SESSION_ID`` the bot
          process inherited.
        * names beginning with ``_``, tmux's hidden-window convention, which
          also covers herdr's ``__…__`` form.

        Without this the bot adopts every session the user has open, its own
        included, and starts typing into terminals nobody pointed it at.
        """
        own = self._own_session_id
        if own and str(session.get("id", "")).casefold() == own:
            return False
        return not str(session.get("name") or "").startswith("_")

    # ── Multiplexer Protocol surface ───────────────────────────────────

    async def ensure_session(self) -> None:
        """Verify the agterm control socket answers.

        agterm owns its own windows and workspaces, so there is nothing to
        create; a reachable socket is the whole precondition.
        """
        if await self._tree() is None:
            raise RuntimeError(
                "agterm is not reachable — is it running, and is AGTERM_SOCKET set?"
            )

    async def list_windows(self) -> list[WindowRef]:
        """List the sessions discovery may adopt (best-effort, [] on failure).

        Narrower than ``list_windows_for_reconciliation`` on purpose: this is
        the listing discovery turns into topics, so it carries the workspace
        scope and the self/hidden exclusions.
        """
        tree = await self._tree()
        if tree is None:
            return []
        return [
            self._to_window(session)
            for workspace, session in self._sessions(tree)
            if self._in_scope(workspace) and self._is_adoptable(session)
        ]

    async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
        """List windows, or None when agterm cannot confirm the listing.

        State cleanup calls this stronger contract instead of ``list_windows``
        so an unreachable socket is never mistaken for "every window is gone"
        and used to prune live state. None means unknown; [] means agterm
        answered and holds no sessions.

        Deliberately unfiltered, unlike ``list_windows``: cleanup asks what
        exists, not what may be adopted. A bound session the user moved to
        another workspace, or renamed with a leading underscore, must not read
        as gone and be pruned.
        """
        tree = await self._tree()
        if tree is None:
            return None
        return [
            self._to_window(session) for _workspace, session in self._sessions(tree)
        ]

    async def list_workspaces(self) -> list[WorkspaceRef]:
        """List agterm workspaces; the id is an opaque token for ``create_window``."""
        tree = await self._tree()
        if tree is None:
            return []
        out: list[WorkspaceRef] = []
        for workspace in tree.get("workspaces", []):
            if not isinstance(workspace, dict) or not workspace.get("id"):
                continue
            # A workspace has no directory of its own in agterm; the neutral
            # type wants one, so report its first session's cwd when there is
            # one and an empty string otherwise.
            sessions = [s for s in workspace.get("sessions", []) if isinstance(s, dict)]
            cwd = str(sessions[0].get("cwd") or "") if sessions else ""
            out.append(
                WorkspaceRef(
                    workspace_id=str(workspace["id"]),
                    label=str(workspace.get("name") or ""),
                    cwd=cwd,
                )
            )
        return out

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        """Find a session by its UUID; None when it no longer exists."""
        session = await self._find_session(window_id)
        return None if session is None else self._to_window(session)

    async def _read_text(self, window_id: str, *, lines: int | None) -> str | None:
        """Read the agent pane's buffer as plain text; None on a failed read.

        No existence preflight: ``session text`` already fails on an unknown
        target, so a probe would double every call on the polling path and
        still leave a window between the check and the read.
        """
        args = ["session", "text", "--pane", _AGENT_PANE, "--target", window_id]
        if lines is not None and lines > 0:
            args += ["--lines", str(lines)]
        result = await self._call(args)
        if result is None:
            return None
        text = result.get("text")
        return text if isinstance(text, str) else None

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the agent pane's visible text.

        ``with_ansi`` is accepted for contract compatibility and ignored: agterm
        returns UTF-8 grid text with no SGR. Spacing, row boundaries and
        box-drawing survive, so structural parsing is unaffected; only rendered
        screenshots lose styling, and the renderer falls back to default styles.
        """
        del with_ansi
        text = await self._read_text(window_id, lines=None)
        return text.strip() or None if text is not None else None

    async def capture_pane_scrollback(
        self, window_id: str, history: int = 200
    ) -> str | None:
        """Capture the agent pane including scrollback (plain text)."""
        text = await self._read_text(window_id, lines=history)
        return text.strip() or None if text is not None else None

    async def capture_scrollback(
        self, window_id: str, lines: int = 200
    ) -> CaptureResult | None:
        """Capture with scrollback; agterm has no line cap so never truncated."""
        text = await self._read_text(window_id, lines=lines)
        if text is None:
            return None
        return CaptureResult(text=text, truncated=False)

    async def capture_pane_by_id(
        self,
        pane_id: str,
        *,
        with_ansi: bool = False,
        window_id: str | None = None,
    ) -> str | None:
        """Reject raw pane locators; only a session id may authorise a read."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw agterm pane read")
            return None
        return await self.capture_pane(window_id, with_ansi=with_ansi)

    async def pane_dims(self, window_id: str) -> PaneDims | None:
        """Always None — agterm reports no pane geometry over the control channel."""
        del window_id
        return None

    async def _type(self, window_id: str, payload: str) -> bool:
        """Inject one payload into the agent pane.

        ``--stdin`` keeps arbitrary control bytes out of argv, where the
        argument parser could mangle them.
        """
        return await self._call_ok(
            [
                "session",
                "type",
                "--stdin",
                "--pane",
                _AGENT_PANE,
                "--target",
                window_id,
            ],
            payload,
        )

    async def _send_literal(
        self, window_id: str, text: str, *, enter: bool, raw: bool = False
    ) -> bool:
        """Type literal text, then submit separately when asked.

        ``raw`` skips the bracketed-paste wrapping, matching what that flag
        means on the tmux backend: no TUI-shaped handling. A shell that never
        enabled DEC mode 2004 echoes the markers as text, and the raw path is
        how a ``!``-prefixed shell command reaches a pane.
        """
        if text:
            wrap = not raw and "\n" in text
            payload = f"{_PASTE_START}{text}{_PASTE_END}" if wrap else text
            if not await self._type(window_id, payload):
                return False
        if not enter:
            return True
        await asyncio.sleep(_SUBMIT_DELAY_SECONDS)
        return await self._type(window_id, "\r")

    async def send(
        self,
        window_id: str,
        text: str,
        *,
        enter: bool = True,
        literal: bool = True,
        raw: bool = False,
    ) -> bool:
        """Inject text or keys into the agent pane.

        ``literal=False`` means *text* is a whitespace-separated run of tmux key
        names, translated here to the bytes a pty expects. ``raw`` is accepted
        for contract compatibility; agterm needs none of the tmux TUI
        workarounds it disables.

        A literal send goes out as two injections, text then Return, with the
        same pause the tmux backend uses: agterm turns every newline it is
        given into a Return keypress, so appending one submits in the same
        keystroke burst that delivered the text, and a TUI still redrawing
        takes the submit before it has the prompt.

        Multi-line text is wrapped in bracketed-paste markers for the same
        reason: without them each interior newline is its own submit, and one
        Telegram message arrives as several partial prompts.
        """
        # As in the read path, ``session type`` fails on an unknown target by
        # itself, so no existence preflight.
        if literal:
            return await self._send_literal(window_id, text, enter=enter, raw=raw)
        chunks: list[str] = []
        for token in text.split():
            resolved = _key_to_bytes(token)
            if resolved is None:
                logger.debug("unknown agterm key name", token=token)
                return False
            chunks.append(resolved)
        if not chunks:
            return False
        if enter:
            chunks.append("\r")
        payload = "".join(chunks)
        return await self._type(window_id, payload)

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
    ) -> bool:
        """Reject raw pane locators; only a session id may authorise a write."""
        if window_id is None or pane_id != window_id:
            logger.warning("Rejected raw agterm pane write")
            return False
        return await self.send(window_id, text, enter=enter, literal=literal)

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
        """Close the session. True on success."""
        return await self._call_ok(["session", "close", "--target", window_id])

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename the session's sidebar label. True on success."""
        return await self._call_ok(
            ["session", "rename", new_name, "--target", window_id]
        )

    async def list_panes(self, window_id: str) -> list[PaneInfo]:
        """Return no pane handles: agterm exposes no durable sibling identity.

        A split pane's ``right`` role is not stable — agterm promotes the split
        survivor into the main pane when the primary exits — so returning it as
        a ``PaneInfo.pane_id`` would hand callers a locator that silently starts
        addressing a different shell.
        """
        del window_id
        return []

    async def split_window(self, window_id: str) -> str | None:
        """Always None — see ``list_panes`` for why there is no safe handle."""
        del window_id
        return None

    @staticmethod
    def _abandon(window_id: str, reason: str, closed: bool) -> str:
        """Compose the failure message, saying so when the cleanup close failed.

        The same outage that fails creation usually fails the close too, and
        the caller never learns the id, so a silent failure leaves a session
        nobody can reach and discovery later adopts.
        """
        if closed:
            return reason
        logger.warning(
            "agterm session left behind after a failed creation", window_id=window_id
        )
        return f"{reason}; the session could not be closed and may still be open"

    async def _await_realized(self, window_id: str) -> bool:
        """Wait until agterm reports the new session's terminal is up.

        ``session new`` answers as soon as the deck holds the session, but the
        surface needs a SwiftUI and an AppKit pass after that, and creation
        while the display sleeps can defer it further. agterm reports the state
        per session as ``realized``; sending into a session before it flips
        loses the keystrokes.
        """
        for _attempt in range(_REALIZE_ATTEMPTS):
            session = await self._find_session(window_id)
            if session is None:
                return False
            if session.get("realized"):
                return True
            await asyncio.sleep(_REALIZE_POLL_SECONDS)
        return False

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
        """Create a shell session and type the agent command into it.

        Mirrors the tmux backend: a login shell first, then the command as
        keystrokes. agterm's ``--command`` looks like the shorter route and is
        the wrong one twice over. It runs the program with the **app's GUI
        PATH**, so a Homebrew or ``~/.local/bin`` agent exits 127; and it
        replaces the login shell, so the session closes the moment the agent
        exits, where a tmux window survives as a shell. Together those turn a
        mistyped or unresolvable command into a session that is created,
        reported as created, and gone a second later.

        Created with ``--no-select`` so an automated creation never steals the
        selection from whoever is at the keyboard.

        The directory is validated here for the same reason the tmux backend
        validates it: agterm accepts an arbitrary cwd and can answer ``ok``
        before the surface fails, so an unusable path would otherwise be
        reported as a created window.
        """
        path, problem = _validated_work_dir(work_dir)
        if path is None:
            return (False, problem, "", "")
        args = ["session", "new", "--cwd", work_dir, "--no-select"]
        if window_name:
            args += ["--name", window_name]
        if workspace_id:
            args += ["--workspace", workspace_id]
        result = await self._call(args)
        if result is None:
            return (False, "agterm could not create the session", "", "")
        window_id = str(result.get("id", ""))
        if not window_id:
            return (False, "agterm returned no session id", "", "")
        # From here the session exists, so every failure closes it before
        # returning. Abandoning it would leave a session the caller cannot
        # clean up (it never learns the id) and which discovery then adopts as
        # a topic of its own.
        if not await self._await_realized(window_id):
            return (
                False,
                self._abandon(
                    window_id,
                    "agterm created the session but its terminal never came up",
                    await self.kill_window(window_id),
                ),
                "",
                "",
            )
        command = launch_command
        if command and agent_args:
            command = f"{command} {agent_args}"
        if start_agent and command and not await self.send(window_id, command):
            closed = await self.kill_window(window_id)
            message = self._abandon(
                window_id, f"could not start {command!r} in the new session", closed
            )
            return (False, message, "", "")
        if window_name:
            # The caller named it, so agterm's auto-basename cannot differ and
            # there is nothing a tree read would tell us.
            return (True, "", window_name, window_id)
        # ``session new`` answers with the id alone, but callers consume the
        # label, so read back the name agterm derived. Fall back to the cwd
        # basename when that read fails, so the label is never empty.
        session = await self._find_session(window_id)
        name = str(session.get("name") or "") if session else ""
        return (True, "", name or path.name, window_id)

    async def create_topic_target(
        self,
        work_dir: str,
        *,
        launch_command: str | None,
        workspace_id: str | None,
        window_name: str | None = None,
        agent_args: str = "",
    ) -> TopicTargetResult:
        """Create a topic target; the session UUID is its durable target."""
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
        """Not supported on agterm (``native_worktrees`` is False)."""
        del repo_path, worktree_path, branch, window_name, launch_command
        return (False, "agterm does not create worktrees natively", "", "")

    async def foreground(self, window_id: str) -> ForegroundInfo | None:
        """Return the agent pane's foreground argv.

        agterm reports the argv but no pid, pgid or tty, so those are zero and
        empty.

        None at an idle shell prompt, and that is agterm's answer rather than a
        failure: it omits ``foreground`` from the node exactly when the pane
        sits at a prompt. The shell provider needs a shell *name* to install
        its prompt marker, and agterm reports none, so that provider does not
        work on this backend. The adapter does not guess one: ``shell_infra``
        matches ``argv[0]`` against ``KNOWN_SHELLS`` and then sends that
        shell's syntax, so a wrong guess types zsh into a fish prompt in a
        terminal somebody is using. Lifting this needs agterm to report the
        shell on an idle node.

        A pgid of 0 has a consequence worth stating plainly:
        ``providers.process_detection.detect_provider_cached`` returns early on
        ``fg.pgid == 0``, so argv classification never runs for this backend.
        Providers named directly still resolve through
        ``WindowRef.pane_current_command``; a provider launched behind a
        wrapper (``node``/``bun``/``npx``) loses the argv fallback until
        detection learns to classify an uncacheable foreground.
        """
        session = await self._find_session(window_id)
        if session is None:
            return None
        argv = session.get("foreground")
        if not isinstance(argv, list) or not argv:
            return None
        return ForegroundInfo(
            pid=0,
            pgid=0,
            argv=[str(part) for part in argv],
            cwd=str(session.get("cwd") or ""),
            tty="",
        )

    async def agent_status(self, window_id: str) -> AgentStatus | None:
        """Always None: ``native_agent_status`` is False for this backend.

        agterm does report a status, but that capability flag currently also
        selects herdr-shaped creation and topic-eligibility behaviour elsewhere
        in ccgram (see the module docstring), so honouring the contract means
        leaving status to terminal scraping until the flag is split upstream.
        """
        del window_id
        return None

    async def watch_events(
        self, window_ids: Sequence[str]
    ) -> AsyncGenerator[MuxEvent, None]:
        """Empty stream — agterm's control channel has no event subscription."""
        del window_ids
        return
        yield  # pragma: no cover - makes this an async generator

    async def get_pane_title(self, window_id: str) -> str:
        """Return the session's terminal title, or '' when it reports none."""
        session = await self._find_session(window_id)
        if session is None:
            return ""
        return str(session.get("title") or "")

    async def stamp_pane_title(self, window_id: str, provider_name: str) -> None:
        """No-op: agterm has no command to set a pane title.

        The title agterm reports is the OSC title the program itself emits, so
        there is nothing to stamp. Provider re-detection falls back to the
        foreground argv, which agterm does report.
        """
        del window_id, provider_name
