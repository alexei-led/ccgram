"""Shell provider — chat-first shell interface via Telegram.

Extends JsonlProvider to inherit default no-op implementations.
Tmux opens the user's $SHELL by default; overrides only what differs
from the base class (no transcripts, no commands, no bash output).
Prompt marker ({prefix}:N❯) enables output isolation and exit code detection.
The prefix is configurable via CCGRAM_PROMPT_MARKER (default "ccgram").
"""

import asyncio
import functools
import os
import re
from typing import Any, ClassVar

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import ProviderCapabilities

_DEFAULT_MARKER = "ccgram"


def _get_marker_prefix() -> str:
    """Return the configured prompt marker prefix."""
    from ccgram.config import config

    return getattr(config, "prompt_marker", _DEFAULT_MARKER) or _DEFAULT_MARKER


@functools.cache
def _compile_prompt_re(prefix: str) -> re.Pattern[str]:
    """Compile prompt regex for a given prefix (cached per unique prefix)."""
    return re.compile(rf"^{re.escape(prefix)}:(\d+)❯\s?(.*)")


def get_prompt_re() -> re.Pattern[str]:
    """Return compiled prompt regex for the configured marker prefix."""
    return _compile_prompt_re(_get_marker_prefix())


KNOWN_SHELLS = frozenset({"bash", "zsh", "fish", "sh", "dash", "tcsh", "csh", "ksh"})


async def has_prompt_marker(window_id: str) -> bool:
    """Check if the prompt marker is present in the pane."""
    from ccgram.tmux_manager import tmux_manager

    capture = await tmux_manager.capture_pane(window_id)
    if not capture:
        return False
    prompt_re = get_prompt_re()
    return any(prompt_re.match(line) for line in capture.rstrip().splitlines()[-5:])


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


async def setup_shell_prompt(window_id: str, *, clear: bool = True) -> None:
    """Override shell prompt with marker including exit code.

    No-op if the marker is already present in the pane (idempotent).
    Set ``clear=False`` when attaching to an existing session to
    preserve scrollback context.
    """
    if await has_prompt_marker(window_id):
        return

    from ccgram.tmux_manager import tmux_manager

    prefix = _get_marker_prefix()
    shell = await detect_pane_shell(window_id)
    cmds = {
        "fish": f'function fish_prompt; printf "{prefix}:$status❯ "; end',
        "bash": f"PS1='{prefix}:$?❯ '",
        "zsh": f"PROMPT='{prefix}:%?❯ '",
        "tcsh": f'set prompt = "{prefix}:$status❯ "',
        "csh": f'set prompt = "{prefix}:$status❯ "',
    }
    cmd = cmds.get(shell, cmds["bash"])
    await tmux_manager.send_keys(window_id, cmd)
    await asyncio.sleep(0.3)
    if clear:
        await tmux_manager.send_keys(window_id, "clear")


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
