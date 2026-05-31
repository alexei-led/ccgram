"""Shell provider — chat-first shell interface via Telegram.

Slim ``AgentProvider`` implementation. The bulk of shell-related infrastructure
(prompt detection, marker setup, ``KNOWN_SHELLS``, ``PromptMatch``, etc.) lives
in ``shell_infra.py``. Names are re-exported here for backward compatibility
with handlers and tests that import from ``ccgram.providers.shell``.

Two prompt modes for output isolation and exit code detection:
- ``wrap`` (default): appends a small ``⌘N⌘`` marker after the user's
  existing prompt, preserving Tide / Starship / Powerlevel10k / etc.
- ``replace``: replaces the entire prompt with ``{prefix}:N❯``
  (the legacy behaviour, opt-in via ``CCGRAM_PROMPT_MODE=replace``).
"""

from typing import Any, ClassVar

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import ProviderCapabilities

# Re-exports for backward compat — new code should import directly from
# ccgram.providers.shell_infra. Listed in __all__ so star-imports continue to
# work and tools see the public surface.
from ccgram.providers.shell_infra import (  # noqa: F401
    KNOWN_SHELLS,
    PromptMatch,
    detect_pane_shell,
    get_shell_name,
    has_prompt_marker,
    match_prompt,
    setup_shell_prompt,
)


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
        chat_first_command_path=True,
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
