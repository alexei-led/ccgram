"""Oh My Pi (``omp``) provider — https://github.com/can1357/oh-my-pi.

Oh My Pi is a fork of pi: the transcript format is the same pi-family JSONL v3
envelope, parsed by ``pi_format`` (roles, ``toolCall``/``toolResult`` blocks,
``stopReason`` errors). What differs is the storage layout and the CLI surface.

Sessions live at ``~/.omp/agent/sessions/<encoded-cwd>/<timestamp>_<uuid>.jsonl``
— the same one-JSONL-per-session shape as pi, but the cwd bucket encoding is
omp's own (home- and tmp-relative prefixes, see ``encode_cwd_dirname``). Every
transcript starts with a ``type: title`` entry *before* the ``type: session``
header, so the header is not line 1 (``pi_format.read_session_header`` scans
past it).

ccgram treats omp as hookless: there is no omp equivalent of the pi
hook-runner extension and ccgram installs no provider hooks, so session
tracking is transcript discovery (``discover_transcript``) and the transcript
file is the message source of truth. ``make_launch_args`` is inherited from
``PiProvider``: resume always uses ``--session <path>`` because ``--resume``
with no value opens an interactive picker ccgram cannot drive over
``send_keys``.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from ccgram.providers.base import (
    DiscoveredCommand,
    SessionStartEvent,
)
from ccgram.providers.omp_discovery import (
    _OMP_TELEGRAM_BUILTINS,
    discover_omp_commands,
)
from ccgram.providers.pi import PiProvider
from ccgram.providers.session_scan import newest_matching_transcript


def _omp_sessions_dir() -> Path:
    return Path.home() / ".omp" / "agent" / "sessions"


# Cap transcript age when the pane is dead — guards against picking up an
# unrelated historical transcript for the same cwd.
_OMP_STALE_TRANSCRIPT_MAX_AGE_SECS = 120.0

# How many recent session files to inspect when searching for a cwd match.
_OMP_DISCOVERY_SCAN_LIMIT = 20


def _resolve(path: str | Path) -> Path:
    """Canonicalize *path*, falling back to the literal path on OSError."""
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _relative_bucket(prefix: str, target: Path, root: Path) -> str | None:
    """Render ``prefix + <target relative to root>``, or None if not under root."""
    try:
        relative = target.relative_to(_resolve(root))
    except ValueError:
        return None
    parts = [part for part in relative.parts if part not in ("", ".")]
    return prefix + "-".join(parts)


def encode_cwd_dirname(cwd: str) -> str:
    """Encode a working directory into omp's session subdirectory name.

    omp canonicalizes the cwd first (so symlink aliases share a bucket), then:

    - under the home directory → ``-`` + home-relative path, separators ``-``
    - under the temp root → ``-tmp-`` + temp-relative path, separators ``-``
    - otherwise → ``--`` + absolute path minus the leading ``/`` + ``--``

    Edges: the home directory itself renders as ``-`` and the temp root as
    ``-tmp-`` (empty relative path), and the filesystem root renders as
    ``----`` (the absolute branch with nothing left to encode). None of them
    raise.
    """
    resolved = _resolve(cwd)

    for prefix, root in (("-", Path.home()), ("-tmp-", tempfile.gettempdir())):
        bucket = _relative_bucket(prefix, resolved, Path(root))
        if bucket is not None:
            return bucket

    return "--" + str(resolved).strip("/").replace("/", "-") + "--"


class OmpProvider(PiProvider):
    """AgentProvider implementation for the Oh My Pi CLI."""

    _CAPS = replace(
        PiProvider._CAPS,
        name="omp",
        launch_command="omp",
        # pi's hook support comes from the third-party hook-runner extension;
        # omp has no such contract and ccgram installs no provider hooks, so
        # every hook-driven path must stay off for it.
        supports_hook=False,
        builtin_commands=tuple(_OMP_TELEGRAM_BUILTINS.keys()),
        # omp's app.message.followUp default is ctrl+q (also ctrl+enter); pi's
        # follow-up key is Alt+Enter, so the key differs between the two.
        followup_key="C-q",
        # Unverified against a live omp TUI (pi's set was verified by driving
        # the real CLI): kept to commands whose registry entry clearly opens an
        # in-TUI menu the user drives with arrows/Enter/Esc.
        tui_picker_commands=frozenset(
            {
                "agents",
                "copy",
                "extensions",
                "login",
                "logout",
                "mcp",
                "model",
                "session",
                "settings",
                "share",
                "switch",
                "todo",
            }
        ),
    )

    _BUILTINS = _OMP_TELEGRAM_BUILTINS

    # ── Discovery ────────────────────────────────────────────────────────

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Return the newest omp transcript whose header cwd matches."""
        age_limit = (
            _OMP_STALE_TRANSCRIPT_MAX_AGE_SECS if max_age is None else float(max_age)
        )
        return newest_matching_transcript(
            _omp_sessions_dir() / encode_cwd_dirname(cwd),
            cwd,
            window_key=window_key,
            max_age=age_limit,
            scan_limit=_OMP_DISCOVERY_SCAN_LIMIT,
        )

    # ── Commands ────────────────────────────────────────────────────────

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        return discover_omp_commands(base_dir)
