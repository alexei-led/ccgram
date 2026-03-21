"""Shell provider — chat-first shell interface via Telegram.

Standalone provider (not extending JsonlProvider — no transcripts).
Tmux opens the user's $SHELL by default; all protocol methods return no-op/None.
"""

from typing import Any

from ccgram.providers.base import (
    AgentMessage,
    DiscoveredCommand,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)


class ShellProvider:
    """AgentProvider implementation for raw shell sessions."""

    _CAPS = ProviderCapabilities(
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

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._CAPS

    def make_launch_args(
        self,
        resume_id: str | None = None,  # noqa: ARG002
        use_continue: bool = False,  # noqa: ARG002
    ) -> str:
        return ""

    def parse_hook_payload(
        self,
        payload: dict[str, Any],  # noqa: ARG002
    ) -> SessionStartEvent | None:
        return None

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

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],  # noqa: ARG002
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        return [], dict(pending_tools)

    def parse_terminal_status(
        self,
        pane_text: str,  # noqa: ARG002
        *,
        pane_title: str = "",  # noqa: ARG002
    ) -> StatusUpdate | None:
        return None

    def extract_bash_output(
        self,
        pane_text: str,  # noqa: ARG002
        command: str,  # noqa: ARG002
    ) -> str | None:
        return None

    def is_user_transcript_entry(
        self,
        entry: dict[str, Any],  # noqa: ARG002
    ) -> bool:
        return False

    def parse_history_entry(
        self,
        entry: dict[str, Any],  # noqa: ARG002
    ) -> AgentMessage | None:
        return None

    def discover_transcript(
        self,
        cwd: str,  # noqa: ARG002
        window_key: str,  # noqa: ARG002
        *,
        max_age: float | None = None,  # noqa: ARG002
    ) -> SessionStartEvent | None:
        return None

    def requires_pane_title_for_detection(
        self,
        pane_current_command: str,  # noqa: ARG002
    ) -> bool:
        return False

    def detect_from_pane_title(
        self,
        pane_current_command: str,  # noqa: ARG002
        pane_title: str,  # noqa: ARG002
    ) -> bool:
        return False

    def discover_commands(
        self,
        base_dir: str,  # noqa: ARG002
    ) -> list[DiscoveredCommand]:
        return []
