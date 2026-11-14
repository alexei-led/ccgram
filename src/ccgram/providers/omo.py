"""omo coding agent provider — omo-ai on top of @code-yeongyu/senpi.

omo is a senpi (pi fork) distribution: it keeps pi's JSONL v3 session format
but stores state under ``~/.omo/agent`` and reports a different TUI title.
Everything non-obvious is therefore inherited: session discovery, the
``--session``/``--continue`` resume flags and the TUI parser all come from
PiProvider; this module only overrides identity, state location and the
detection hints a window started as a JS interpreter needs.

Session path: ``~/.omo/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl``

State directory resolution mirrors omo itself
(``omo-ai/bin/lib/agent-dir.js``): ``OMO_CODING_AGENT_DIR`` →
``SENPI_CODING_AGENT_DIR`` → ``PI_CODING_AGENT_DIR`` → ``~/.omo/agent``.
``OMO_AGENT_DIR`` is *not* part of that chain — it addresses the omo daemon,
not the session store.

Detection: when omo is launched from its bin shim, tmux reports
``pane_current_command=bun`` (the CLI is JS). The basename tables therefore
cannot name it; the pane title set by omo's TUI (``OmO - <project>``) can.
``requires_pane_title_for_detection`` asks the caller to fetch the title for
interpreter processes and ``detect_from_pane_title`` performs the match.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from ccgram.providers.base import SessionStartEvent
from ccgram.providers.pi import PiProvider, encode_cwd_dirname, provider_sessions_dir

# Mirrors omo's own resolution order (omo-ai/bin/lib/agent-dir.js).
_AGENT_DIR_ENV = "OMO_CODING_AGENT_DIR"
_AGENT_DIR_FALLBACKS = ("SENPI_CODING_AGENT_DIR", "PI_CODING_AGENT_DIR")

#: Interpreters that can host a JS agent CLI.  A pane whose foreground
#: process is one of these needs the pane title to tell the agents apart.
_INTERPRETER_BASENAMES = frozenset({"bun", "bunx", "node", "npx"})

#: omo's TUI window title starts with this marker (observed: "OmO - <project>").
_TITLE_PREFIX = "omo"

#: How far up the process tree to look for the pane's shell when matching a
#: session holder; omo runs as a direct child of the pane shell.
_MAX_PARENT_HOPS = 8

#: Clock skew allowance when comparing a holder's recorded process start with
#: the live process start; anything larger means the pid was reused.
_START_TIME_TOLERANCE_MS = 5000


def _pane_pids(window_key: str) -> set[int]:
    """Return the pane pids of ``window_key`` (``ccgram:@N``), empty on failure."""
    window_id = window_key.rsplit(":", 1)[-1]
    try:
        # Lazy: the multiplexer is wired by bootstrap and absent in imports.
        from ccgram.multiplexer import get_multiplexer

        session = get_multiplexer("tmux").get_session()
        window = (
            session.windows.get(window_id=window_id, default=None) if session else None
        )
    except (RuntimeError, OSError, AttributeError):  # fmt: skip
        return set()
    pids: set[int] = set()
    for pane in window.panes if window is not None else []:
        try:
            pids.add(int(pane.pane_pid))
        except (TypeError, ValueError):  # fmt: skip
            continue
    return pids


def _holder_pid_map(sessions_dir: Path) -> dict[int, str]:
    """Map pid → session id from omo's ``session-holders/<id>/<pid>.json``.

    Holder files outlive their process, and Linux reuses pids, so an entry is
    only trusted when the live process started when the holder says it did.
    """
    holders: dict[int, str] = {}
    try:
        entries = sessions_dir.glob("*/session-holders/*/*.json")
    except OSError:
        return holders
    for entry in entries:
        try:
            payload = json.loads(entry.read_text())
            pid = int(payload.get("pid", 0))
            recorded_start = int(payload.get("processStartedAtMs", 0))
        except (OSError, ValueError, TypeError):  # fmt: skip
            continue
        if pid <= 0:
            continue
        live_start = _process_started_ms(pid)
        if live_start is None:
            continue
        if (
            recorded_start
            and abs(live_start - recorded_start) > _START_TIME_TOLERANCE_MS
        ):
            continue
        holders[pid] = entry.parent.name
    return holders


def _process_started_ms(pid: int) -> int | None:
    """Return when ``pid`` started, in ms since the epoch, or None if gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        boot = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime")
        )
    except (OSError, ValueError, StopIteration):  # fmt: skip
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    try:
        start_ticks = int(fields[19])
    except (IndexError, ValueError):  # fmt: skip
        return None
    return boot * 1000 + int(start_ticks * 1000 / os.sysconf("SC_CLK_TCK"))


def _descends_from(pid: int, ancestors: set[int]) -> bool:
    """Return whether ``pid`` or one of its parents is in ``ancestors``."""
    current = pid
    for _ in range(_MAX_PARENT_HOPS):
        if current in ancestors:
            return True
        try:
            status = Path(f"/proc/{current}/status").read_text()
        except OSError:
            return False
        parent = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        if parent <= 1 or parent == current:
            return False
        current = parent
    return False


class OmoProvider(PiProvider):
    """AgentProvider implementation for the omo coding agent CLI."""

    def _sessions_dir(self) -> Path:
        return provider_sessions_dir(
            (_AGENT_DIR_ENV, *_AGENT_DIR_FALLBACKS), (".omo", "agent")
        )

    _CAPS = replace(
        PiProvider._CAPS,
        name="omo",
        launch_command="omo",
        # omo ships no cc-thingz hook runner; the transcript is the only
        # status source, exactly like a hookless pi.
        supports_hook=False,
        hook_install_managed_by_ccgram=False,
        # `/model` in omo prints a list into the transcript instead of
        # opening a modal picker, so advertising it would offer a picker that
        # cannot be driven.  The owner only needs the default model anyway.
        tui_picker_commands=frozenset(
            name for name in PiProvider._CAPS.tui_picker_commands if name != "model"
        ),
    )

    # ── Exact window → session binding ───────────────────────────────────

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Bind the window to the session its own process holds.

        Newest-file discovery is ambiguous whenever another omo process writes
        into the same cwd (probes, cron runs, a second window): the newest
        transcript may belong to nobody in this window, and the topic then
        tracks a file that never grows.  omo records ownership as
        ``session-holders/<session-id>/<pid>.json``, so the pane's process tree
        names the session exactly.  Fall back to the inherited scan when the
        holder is absent (older omo builds, foreign launchers).
        """
        exact = self._session_for_window(cwd, window_key)
        if exact is not None:
            return exact
        return super().discover_transcript(cwd, window_key, max_age=max_age)

    def _session_for_window(
        self, cwd: str, window_key: str
    ) -> SessionStartEvent | None:
        """Return the session held by this window's process tree, if any."""
        panes = _pane_pids(window_key)
        if not panes:
            return None
        sessions_dir = self._sessions_dir()
        session_dir = sessions_dir / encode_cwd_dirname(cwd)
        for pid, session_id in _holder_pid_map(sessions_dir).items():
            if not _descends_from(pid, panes):
                continue
            matches = sorted(session_dir.glob(f"*{session_id}.jsonl"))
            if not matches:
                continue
            return SessionStartEvent(
                session_id=session_id,
                cwd=cwd,
                transcript_path=str(matches[-1]),
                window_key=window_key,
            )
        return None

    # ── Detection hints ──────────────────────────────────────────────────

    @staticmethod
    def _is_interpreter(pane_current_command: str) -> bool:
        command = pane_current_command.strip().lower()
        if not command:
            return False
        basename = command.split()[0].rsplit("/", 1)[-1]
        return basename in _INTERPRETER_BASENAMES

    def requires_pane_title_for_detection(self, pane_current_command: str) -> bool:
        """A bun/node pane may be pi, omo or something else — ask the title."""
        return self._is_interpreter(pane_current_command)

    def detect_from_pane_title(
        self, pane_current_command: str, pane_title: str
    ) -> bool:
        """Match omo's TUI title (``OmO - <project>``) on interpreter panes."""
        if not self._is_interpreter(pane_current_command):
            return False
        title = " ".join(pane_title.split()).lower()
        return title.startswith(_TITLE_PREFIX)
