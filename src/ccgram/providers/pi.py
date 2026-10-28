"""Pi coding agent provider — https://pi.dev behind AgentProvider.

Pi is a Node.js-based CLI with JSONL session transcripts (v3 format). With the
cc-thingz hook-runner extension, lifecycle hooks provide instant status signals;
transcript discovery remains the fallback and message source of truth. Discovery
scans ``~/.pi/agent/sessions/`` for the newest transcript whose header ``cwd``
matches the window working directory.

Session path:  ``~/.pi/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl``

CWD encoding strips leading slashes, replaces ``/``, ``\\``, and ``:`` with
``-``, then wraps with ``--`` delimiters.  The canonical session id lives in
the header line ``{"type":"session","id":"<uuid>","cwd":"...","version":3}``
— the filename prefix is just a timestamp.

Resume strategy: always use ``--session <path>``.  ``--resume`` opens pi's
interactive picker, which ccgram can't drive over ``send_keys``.  Pi accepts
both a transcript path and a (partial) UUID for ``--session``, so we route
everything through it after ``shlex.quote`` for safety.
"""

from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    DiscoveredCommand,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.pi_discovery import _PI_TELEGRAM_BUILTINS, discover_pi_commands
from ccgram.providers.pi_format import read_session_header

from ccgram.providers.pi_tui import (
    PI_QUESTION_FOOTER as PI_QUESTION_FOOTER,
    parse_pi_ui,
)


def _pi_sessions_dir() -> Path:
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    root = Path(agent_dir) if agent_dir else Path.home() / ".pi" / "agent"
    return root / "sessions"


# Cap transcript age when the pane is dead — guards against picking up an
# unrelated historical transcript for the same cwd.
_STALE_TRANSCRIPT_MAX_AGE_SECS = 120.0

# How many recent session files to inspect when searching for a cwd match.
_DISCOVERY_SCAN_LIMIT = 20


def encode_cwd_dirname(cwd: str) -> str:
    """Encode a working directory into pi's session subdirectory name.

    Matches pi's own encoding: strip leading slash, then replace ``/``, ``\\``
    and ``:`` with ``-``, then wrap with ``--``.  Root (``/``) renders as
    ``----`` to stay round-trippable.
    """
    stripped = cwd.lstrip("/\\")
    # Also drop trailing separators so "/tmp/foo/" and "/tmp/foo" collide correctly.
    stripped = stripped.rstrip("/\\")
    encoded = stripped.replace("/", "-").replace("\\", "-").replace(":", "-")
    return f"--{encoded}--"


def _candidate_transcripts(cwd: str) -> list[tuple[float, Path]]:
    """Return ``(mtime, path)`` tuples for this cwd's sessions, newest first."""
    session_dir = _pi_sessions_dir() / encode_cwd_dirname(cwd)
    if not session_dir.is_dir():
        return []
    results: list[tuple[float, Path]] = []
    try:
        for entry in session_dir.iterdir():
            if entry.suffix == ".jsonl" and entry.is_file():
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                results.append((mtime, entry))
    except OSError:
        return []
    results.sort(key=lambda pair: pair[0], reverse=True)
    return results


class PiProvider(JsonlProvider):
    """AgentProvider implementation for the pi coding agent CLI."""

    _CAPS = ProviderCapabilities(
        name="pi",
        launch_command="pi",
        supports_hook=True,
        hook_install_managed_by_ccgram=True,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=True,
        builtin_commands=tuple(_PI_TELEGRAM_BUILTINS.keys()),
        supports_user_command_discovery=False,
        supports_status_snapshot=False,
        # Pickers verified by sending each command to a live pi process and
        # inspecting tmux capture-pane output (see PR #93 follow-up).
        tui_picker_commands=frozenset(
            {
                "model",
                "thinking",
                "login",
                "logout",
                "fork",
                "clone",
                "scoped-models",
                "settings",
                "tree",
            }
        ),
    )

    _BUILTINS = _PI_TELEGRAM_BUILTINS

    # ── Launch ───────────────────────────────────────────────────────────

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build CLI args.  Prefers ``--session`` (non-interactive)."""
        if resume_id:
            return f"--session {shlex.quote(resume_id)}"
        if use_continue:
            return "--continue"
        return ""

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002 — protocol signature
    ) -> StatusUpdate | None:
        """Recognize Pi's current selectors and cc-thingz question menu."""
        return parse_pi_ui(pane_text)

    # ── Discovery ────────────────────────────────────────────────────────

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Return the newest pi transcript whose header cwd matches."""
        if not cwd:
            return None

        age_limit = (
            _STALE_TRANSCRIPT_MAX_AGE_SECS if max_age is None else float(max_age)
        )
        now = time.time()
        try:
            resolved_target = str(Path(cwd).resolve())
        except OSError:
            return None

        for mtime, path in _candidate_transcripts(cwd)[:_DISCOVERY_SCAN_LIMIT]:
            if age_limit > 0 and now - mtime > age_limit:
                break
            header = read_session_header(str(path))
            if not header:
                continue
            try:
                header_cwd = str(Path(header["cwd"]).resolve())
            except OSError:
                continue
            if header_cwd != resolved_target:
                continue
            return SessionStartEvent(
                session_id=header["id"],
                cwd=header["cwd"],
                transcript_path=str(path),
                window_key=window_key,
            )
        return None

    # ── Commands ─────────────────────────────────────────────────────────

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        return discover_pi_commands(base_dir)
