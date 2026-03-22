"""Shell provider — chat-first shell interface via Telegram.

Extends JsonlProvider to inherit default no-op implementations.
Tmux opens the user's $SHELL by default; overrides only what differs
from the base class (no transcripts, no commands, no bash output).
Prompt marker (ccgram:N❯) enables output isolation and exit code detection.
"""

import asyncio
import os
import re
from typing import Any, ClassVar

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import ProviderCapabilities

PROMPT_MARKER = "ccgram:"
PROMPT_RE = re.compile(r"^ccgram:(\d+)❯\s?(.*)")


async def has_prompt_marker(window_id: str) -> bool:
    """Check if the ccgram prompt marker is present in the pane."""
    from ccgram.tmux_manager import tmux_manager

    capture = await tmux_manager.capture_pane(window_id)
    return bool(capture and PROMPT_MARKER in capture)


async def setup_shell_prompt(window_id: str) -> None:
    """Override shell prompt with ccgram marker including exit code."""
    from ccgram.tmux_manager import tmux_manager

    shell = os.environ.get("SHELL", "").rsplit("/", 1)[-1]
    cmds = {
        "fish": 'function fish_prompt; printf "ccgram:$status❯ "; end',
        "bash": "PS1='ccgram:$?❯ '",
        "zsh": "PROMPT='ccgram:%?❯ '",
    }
    cmd = cmds.get(shell, cmds["bash"])
    await tmux_manager.send_keys(window_id, cmd)
    await asyncio.sleep(0.3)
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
