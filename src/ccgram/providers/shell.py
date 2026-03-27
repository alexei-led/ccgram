"""Shell provider — chat-first shell interface via Telegram.

Extends JsonlProvider to inherit default no-op implementations.
Tmux opens the user's $SHELL by default; overrides only what differs
from the base class (no transcripts, no commands, no bash output).

Two prompt modes for output isolation and exit code detection:
- ``wrap`` (default): appends a small ``⌘N⌘`` marker after the user's
  existing prompt, preserving Tide / Starship / Powerlevel10k / etc.
- ``replace``: replaces the entire prompt with ``{prefix}:N❯``
  (the legacy behaviour, opt-in via ``CCGRAM_PROMPT_MODE=replace``).
"""

import asyncio
import functools
import os
import re
from typing import Any, ClassVar

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import ProviderCapabilities

_DEFAULT_MARKER = "ccgram"


_VALID_PROMPT_MODES = frozenset({"wrap", "replace"})
_WARNED_INVALID_MODE = False


def _get_prompt_mode() -> str:
    """Return the configured prompt mode (``wrap`` or ``replace``)."""
    global _WARNED_INVALID_MODE  # noqa: PLW0603
    from ccgram.config import config

    mode = getattr(config, "prompt_mode", "wrap") or "wrap"
    if mode not in _VALID_PROMPT_MODES:
        if not _WARNED_INVALID_MODE:
            _WARNED_INVALID_MODE = True
            import structlog

            structlog.get_logger().warning(
                "Invalid CCGRAM_PROMPT_MODE=%r, defaulting to 'wrap'", mode
            )
        return "wrap"
    return mode


def _get_marker_prefix() -> str:
    """Return the configured prompt marker prefix (used in ``replace`` mode)."""
    from ccgram.config import config

    return getattr(config, "prompt_marker", _DEFAULT_MARKER) or _DEFAULT_MARKER


@functools.cache
def _compile_replace_re(prefix: str) -> re.Pattern[str]:
    """Compile prompt regex for ``replace`` mode (cached per unique prefix)."""
    return re.compile(rf"^{re.escape(prefix)}:(\d+)❯\s?(.*)")


_WRAP_RE = re.compile(r"⌘(\d+)⌘\s?(.*)$")


def match_prompt(line: str) -> re.Match[str] | None:
    """Match a prompt marker in *line*, respecting the current prompt mode.

    In ``replace`` mode the marker is at line start (``re.match``).
    In ``wrap`` mode the marker can appear anywhere (``re.search``).
    """
    if _get_prompt_mode() == "replace":
        return _compile_replace_re(_get_marker_prefix()).match(line)
    return _WRAP_RE.search(line)


KNOWN_SHELLS = frozenset({"bash", "zsh", "fish", "sh", "dash", "tcsh", "csh", "ksh"})


async def has_prompt_marker(window_id: str) -> bool:
    """Check if the prompt marker is present in the pane."""
    from ccgram.tmux_manager import tmux_manager

    capture = await tmux_manager.capture_pane(window_id)
    if not capture:
        return False
    return any(match_prompt(line) for line in capture.rstrip().splitlines()[-5:])


def get_shell_name() -> str:
    """Return the basename of the bot process's $SHELL (e.g. 'fish', 'zsh').

    Sync fallback — for pane-accurate detection use ``detect_pane_shell()``.
    """
    return os.environ.get("SHELL", "").rsplit("/", 1)[-1]


async def detect_pane_shell(window_id: str) -> str:
    """Detect the shell running in a tmux pane via pane_current_command.

    Falls back to ``get_shell_name()`` when the pane is unavailable or
    its command is not a recognized shell.
    """
    from ccgram.tmux_manager import tmux_manager

    window = await tmux_manager.find_window_by_id(window_id)
    if window and window.pane_current_command:
        tokens = window.pane_current_command.split()
        if not tokens:
            return get_shell_name()
        basename = os.path.basename(tokens[0])
        cleaned = basename.lstrip("-")
        if cleaned in KNOWN_SHELLS:
            return cleaned
    return get_shell_name()


def _wrap_setup_commands(shell: str) -> str:
    """Return the shell command that appends a ⌘N⌘ marker to the prompt."""
    # Fish: wrap existing fish_prompt, preserving Tide/Starship/etc.
    # Uses set_color instead of raw ANSI — avoids escape mangling via send_keys.
    # Guard: skip if __ccgram_orig_prompt already exists (idempotent).
    # Embedded clear hides the setup command from the user.
    fish = (
        "functions --query __ccgram_orig_prompt; or begin; "
        "functions --copy fish_prompt __ccgram_orig_prompt 2>/dev/null; "
        "or function __ccgram_orig_prompt; end; "
        "function fish_prompt; "
        "set -l __s $status; "
        "__ccgram_orig_prompt; "
        "set_color brblack; printf '⌘%d⌘ ' $__s; set_color normal; "
        "end; clear; end"
    )
    # Bash: save exit code in PROMPT_COMMAND before user hooks run,
    # then append marker to existing PS1.  Guard: skip if __ccgram_sc exists.
    bash = (
        "type __ccgram_sc >/dev/null 2>&1 || { "
        "__ccgram_sc(){ __ccgram_x=$?; return $__ccgram_x; }; "
        'PROMPT_COMMAND="__ccgram_sc${PROMPT_COMMAND:+;$PROMPT_COMMAND}"; '
        'PS1="${PS1}\\[\\033[2m\\]⌘\\${__ccgram_x}⌘\\[\\033[0m\\] "; '
        "clear; }"
    )
    # Zsh: append marker to existing PROMPT.  Guard: skip if marker present.
    # \\? in the glob escapes ? to match literal (zsh expands %? at render time).
    zsh = (
        '[[ "$PROMPT" == *⌘%\\?⌘* ]] || { '
        'PROMPT="${PROMPT}%{\\033[2m%}⌘%?⌘%{\\033[0m%} "; '
        "clear; }"
    )
    # tcsh/csh: append marker to existing prompt (no dim support).
    # No inline guard — tcsh lacks POSIX case/block syntax for one-liners.
    # The Python-level has_prompt_marker() check provides idempotency.
    tcsh = 'set prompt = "${prompt}⌘$status⌘ "'
    # POSIX sh/dash/ksh: replace prompt (can't reliably wrap).
    # Static ⌘0⌘ marker — POSIX sh doesn't expand $? in PS1 dynamically.
    sh = 'case "$PS1" in *⌘*⌘*) ;; *) PS1="\\$ ⌘0⌘ "; clear;; esac'
    return {
        "fish": fish,
        "bash": bash,
        "zsh": zsh,
        "tcsh": tcsh,
        "csh": tcsh,
        "sh": sh,
        "dash": sh,
        "ksh": sh,
    }.get(shell, sh)


def _replace_setup_commands(shell: str, prefix: str) -> str:
    """Return the shell command that replaces the prompt with {prefix}:N❯."""
    cmds = {
        "fish": f'function fish_prompt; printf "{prefix}:$status❯ "; end',
        "bash": f"PS1='{prefix}:$?❯ '",
        "zsh": f"PROMPT='{prefix}:%?❯ '",
        "tcsh": f'set prompt = "{prefix}:$status❯ "',
        "csh": f'set prompt = "{prefix}:$status❯ "',
    }
    return cmds.get(shell, cmds["bash"])


async def _is_interactive_shell(window_id: str) -> bool:
    """Check if the pane has an interactive shell at a prompt (not running a script).

    Uses ``ps -t`` to inspect the foreground process. A shell running a script
    (e.g. ``bash ./scripts/restart.sh``) has child processes in the foreground
    group, while an idle interactive shell is its own foreground leader with
    bare args like ``-bash``, ``fish``, or ``/bin/zsh``.

    Returns True if the shell looks interactive, False if it's running a script
    or if detection fails (fail-safe: don't send C-c to unknown targets).
    """
    from ccgram.tmux_manager import tmux_manager

    w = await tmux_manager.find_window_by_id(window_id)
    if not w or not w.pane_tty:
        return False

    from .process_detection import get_foreground_args

    args, _ = await get_foreground_args(w.pane_tty)
    if not args:
        return False

    # Interactive shells have bare args: -bash, fish, /usr/bin/zsh, etc.
    # Script-running shells have: bash ./script.sh, bash -c '...', etc.
    first_token = args.split()[0]
    basename = first_token.rsplit("/", 1)[-1].lstrip("-")
    if basename not in KNOWN_SHELLS:
        return False

    # If there are args beyond the shell name, it's running a script/command
    tokens = args.split()
    return len(tokens) == 1


async def setup_shell_prompt(window_id: str, *, clear: bool = True) -> None:
    """Configure the shell prompt with a detectable marker.

    In ``wrap`` mode the existing prompt is preserved and a small ``⌘N⌘``
    suffix is appended.  In ``replace`` mode the prompt is fully replaced
    with ``{prefix}:N❯``.

    No-op if the marker is already present in the pane (idempotent).
    Set ``clear=False`` when attaching to an existing session to
    preserve scrollback context.
    """
    from ccgram.config import config

    # Never send prompt setup to ccgram's own window — the C-c would kill the bot
    if config.own_window_id and window_id == config.own_window_id:
        return

    # Safety: verify the shell is actually idle at a prompt, not running a script.
    # Sending C-c to a shell running restart.sh/ccgram would kill the service.
    if not await _is_interactive_shell(window_id):
        return

    if await has_prompt_marker(window_id):
        return

    from ccgram.tmux_manager import tmux_manager

    # Cancel any partial input to prevent concatenation with the setup command
    await tmux_manager.send_keys(window_id, "C-c", enter=False, literal=False)
    await asyncio.sleep(0.1)

    shell = await detect_pane_shell(window_id)
    mode = _get_prompt_mode()
    if mode == "replace":
        cmd = _replace_setup_commands(shell, _get_marker_prefix())
    else:
        cmd = _wrap_setup_commands(shell)
    await tmux_manager.send_keys(window_id, cmd, raw=True)
    await asyncio.sleep(0.3)
    if clear:
        await tmux_manager.send_keys(window_id, "clear", raw=True)


class ShellProvider(JsonlProvider):
    """AgentProvider implementation for raw shell sessions."""

    _CAPS: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        name="shell",
        launch_command="",
        supports_hook=False,
        supports_hook_events=False,
        supports_resume=False,
        supports_continue=False,
        supports_structured_transcript=False,
        supports_incremental_read=False,
        transcript_format="plain",
    )

    def make_launch_args(
        self,
        resume_id: str | None = None,  # noqa: ARG002
        use_continue: bool = False,  # noqa: ARG002
    ) -> str:
        return ""

    def parse_transcript_line(
        self,
        line: str,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return None

    def read_transcript_file(
        self,
        file_path: str,  # noqa: ARG002
        last_offset: int,  # noqa: ARG002
    ) -> tuple[list[dict[str, Any]], int]:
        return [], 0

    def extract_bash_output(
        self,
        pane_text: str,  # noqa: ARG002
        command: str,  # noqa: ARG002
    ) -> str | None:
        return None
