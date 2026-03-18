"""Contract tests for the AgentProvider protocol.

Every provider must pass these tests. The PROVIDER_FIXTURES list contains
StubProvider plus all real providers (Claude, Codex, Gemini).
"""

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from ccgram.providers.base import (
    AgentMessage,
    AgentProvider,
    DiscoveredCommand,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.providers.gemini import GeminiProvider

# ── Stub provider (minimal conforming implementation) ────────────────────


class StubProvider(JsonlProvider):
    """Minimal provider that satisfies AgentProvider for contract testing.

    Extends JsonlProvider with hook support and continue flag to exercise
    all contract test branches that real hookless providers skip.
    """

    _CAPS = ProviderCapabilities(
        name="stub",
        launch_command="stub-cli",
        supports_hook=True,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        transcript_format="jsonl",
        builtin_commands=("help", "clear"),
    )

    _BUILTINS = {"help": "Show help", "clear": "Clear screen"}

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        if resume_id and self._CAPS.supports_resume:
            return f"--resume {resume_id}"
        if use_continue and self._CAPS.supports_continue:
            return "--continue"
        return ""

    def parse_hook_payload(self, payload: dict[str, Any]) -> SessionStartEvent | None:
        sid = payload.get("session_id", "")
        cwd = payload.get("cwd", "")
        if not sid or not cwd:
            return None
        return SessionStartEvent(
            session_id=sid,
            cwd=cwd,
            transcript_path=payload.get("transcript_path", ""),
            window_key=payload.get("window_key", ""),
        )


# ── Fixtures ─────────────────────────────────────────────────────────────

PROVIDER_FIXTURES: list[type] = [
    StubProvider,
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
]


@pytest.fixture(params=PROVIDER_FIXTURES, ids=lambda cls: cls.__name__)
def provider(request: pytest.FixtureRequest) -> AgentProvider:
    return request.param()


# ── Contract tests ───────────────────────────────────────────────────────


class TestAgentProviderCapabilities:
    def test_required_fields(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        assert caps.name
        assert caps.launch_command

    def test_immutability(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        with pytest.raises(FrozenInstanceError):
            caps.name = "hacked"  # type: ignore[misc]


class TestMakeLaunchArgs:
    def test_fresh_session_returns_empty(self, provider: AgentProvider) -> None:
        result = provider.make_launch_args()
        assert result == ""

    def test_resume_id_included_when_supported(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        resume_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        result = provider.make_launch_args(resume_id=resume_id)
        if caps.supports_resume:
            assert resume_id in result
        else:
            assert resume_id not in result

    def test_continue_when_supported(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        result = provider.make_launch_args(use_continue=True)
        if caps.supports_continue:
            assert result != ""  # Each provider has its own continue syntax
        else:
            assert result == ""


class TestParseHookPayload:
    def test_valid_payload_returns_event(self, provider: AgentProvider) -> None:
        payload = {
            "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "cwd": "/tmp/test",
            "transcript_path": "/tmp/test.jsonl",
            "window_key": "ccgram:@0",
        }
        event = provider.parse_hook_payload(payload)
        if provider.capabilities.supports_hook:
            assert event is not None
            assert isinstance(event, SessionStartEvent)
            assert event.session_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            assert event.cwd == "/tmp/test"
        else:
            assert event is None

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"session_id": ""},
            {"session_id": "x"},
        ],
        ids=["empty", "empty_sid", "invalid_sid_no_cwd"],
    )
    def test_invalid_payload_returns_none(
        self, provider: AgentProvider, payload: dict[str, Any]
    ) -> None:
        assert provider.parse_hook_payload(payload) is None


class TestParseTranscriptLine:
    @pytest.mark.parametrize(
        "line",
        ["", "   ", "not json at all"],
        ids=["empty", "whitespace", "invalid"],
    )
    def test_invalid_returns_none(self, provider: AgentProvider, line: str) -> None:
        assert provider.parse_transcript_line(line) is None

    def test_valid_returns_dict(self, provider: AgentProvider) -> None:
        line = json.dumps({"type": "assistant", "message": {"content": "hi"}})
        result = provider.parse_transcript_line(line)
        assert isinstance(result, dict)
        assert result["type"] == "assistant"


def _make_assistant_entry(
    provider: AgentProvider, text: str = "hello"
) -> dict[str, Any]:
    """Build an assistant transcript entry in the correct format for the provider."""
    name = provider.capabilities.name
    if name == "codex":
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }
    if name == "gemini":
        return {"type": "gemini", "content": text}
    # Claude / stub — standard JSONL
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _make_tool_use_entry(provider: AgentProvider) -> dict[str, Any]:
    """Build a tool_use transcript entry in the correct format."""
    name = provider.capabilities.name
    if name == "codex":
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "t1",
            },
        }
    if name == "gemini":
        return {
            "type": "gemini",
            "content": "Using tool",
            "toolCalls": [{"id": "t1", "name": "Read"}],
        }
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]
        },
    }


def _make_tool_result_entry(provider: AgentProvider) -> dict[str, Any]:
    """Build a tool_result transcript entry in the correct format."""
    name = provider.capabilities.name
    if name == "codex":
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "t1",
                "output": "Chunk ID: 1\nOutput:\nok\n",
            },
        }
    if name == "gemini":
        # Gemini doesn't have explicit tool_result entries — tool calls are
        # tracked in pending but never cleared by a result entry.
        return {"type": "gemini", "content": "result ok"}
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
        },
    }


class TestParseTranscriptEntries:
    def test_empty_returns_empty(self, provider: AgentProvider) -> None:
        messages, pending = provider.parse_transcript_entries([], {})
        assert messages == []
        assert isinstance(pending, dict)

    def test_message_fields(self, provider: AgentProvider) -> None:
        entries = [_make_assistant_entry(provider, "hello")]
        messages, _ = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, AgentMessage)
        assert msg.text == "hello"
        assert msg.role == "assistant"

    def test_pending_carry_over(self, provider: AgentProvider) -> None:
        entries = [_make_tool_use_entry(provider)]
        _, pending = provider.parse_transcript_entries(entries, {})
        assert "t1" in pending

    def test_pending_resolved_on_result(self, provider: AgentProvider) -> None:
        entries = [
            _make_tool_use_entry(provider),
            _make_tool_result_entry(provider),
        ]
        _, pending = provider.parse_transcript_entries(entries, {})
        # Gemini doesn't have explicit tool_result entries
        if provider.capabilities.name == "gemini":
            assert "t1" in pending
        else:
            assert "t1" not in pending


class TestParseTerminalStatus:
    def test_empty_returns_none(self, provider: AgentProvider) -> None:
        assert provider.parse_terminal_status("", pane_title="") is None

    def test_status_update_fields(self, provider: AgentProvider) -> None:
        # Claude detects spinner format; hookless providers (Codex, Gemini)
        # return None for non-interactive panes (no spinner support).
        sep = "─" * 30
        pane = f"output\n✻ Reading files\n{sep}\n❯ \n{sep}\n"
        result = provider.parse_terminal_status(pane, pane_title="")
        if provider.capabilities.name == "claude":
            assert result is not None
            assert isinstance(result, StatusUpdate)
            assert isinstance(result.raw_text, str)
            assert isinstance(result.display_label, str)
        else:
            assert result is None

    def test_plain_text_not_interactive(self, provider: AgentProvider) -> None:
        sep = "─" * 30
        pane = f"output\n✻ Reading files\n{sep}\n❯ \n{sep}\n"
        result = provider.parse_terminal_status(pane, pane_title="")
        if provider.capabilities.name == "claude":
            assert result is not None
            assert result.is_interactive is False
        else:
            assert result is None


class TestExtractBashOutput:
    def test_returns_none_for_empty(self, provider: AgentProvider) -> None:
        assert provider.extract_bash_output("", "ls") is None

    def test_returns_none_when_command_not_found(self, provider: AgentProvider) -> None:
        assert provider.extract_bash_output("some text\nno command here", "ls") is None

    def test_returns_output_when_command_found(self, provider: AgentProvider) -> None:
        pane = "some text\n! ls -la\ntotal 42\n"
        result = provider.extract_bash_output(pane, "ls")
        assert result is not None
        assert result.startswith("! ls")


class TestIsUserTranscriptEntry:
    def test_user_entry_detected(self, provider: AgentProvider) -> None:
        name = provider.capabilities.name
        if name == "codex":
            entry = {"type": "input_item", "payload": {"role": "user"}}
        elif name == "gemini":
            entry = {"type": "user"}
        else:
            entry = {"type": "user"}
        assert provider.is_user_transcript_entry(entry) is True

    def test_non_user_not_detected(self, provider: AgentProvider) -> None:
        name = provider.capabilities.name
        if name == "codex":
            entry = {"type": "response_item", "payload": {"role": "assistant"}}
        elif name == "gemini":
            entry = {"type": "gemini"}
        else:
            entry = {"type": "assistant"}
        assert provider.is_user_transcript_entry(entry) is False

    def test_empty_not_detected(self, provider: AgentProvider) -> None:
        assert provider.is_user_transcript_entry({}) is False


class TestParseHistoryEntry:
    def test_non_message_returns_none(self, provider: AgentProvider) -> None:
        assert provider.parse_history_entry({"type": "summary"}) is None

    def test_assistant_message_parsed(self, provider: AgentProvider) -> None:
        entry = _make_assistant_entry(provider, "hello world")
        result = provider.parse_history_entry(entry)
        assert result is not None
        assert isinstance(result, AgentMessage)
        assert result.role == "assistant"
        assert result.text == "hello world"

    def test_user_message_parsed(self, provider: AgentProvider) -> None:
        name = provider.capabilities.name
        if name == "codex":
            entry = {
                "type": "input_item",
                "payload": {"role": "user", "content": "my question"},
            }
        elif name == "gemini":
            entry = {"type": "user", "content": "my question"}
        else:
            entry = {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "my question"}]},
            }
        result = provider.parse_history_entry(entry)
        assert result is not None
        assert isinstance(result, AgentMessage)
        assert result.role == "user"
        assert result.text == "my question"

    def test_empty_content_returns_none(self, provider: AgentProvider) -> None:
        name = provider.capabilities.name
        if name == "codex":
            entry = {
                "type": "response_item",
                "payload": {"role": "assistant", "content": []},
            }
        elif name == "gemini":
            entry = {"type": "gemini", "content": ""}
        else:
            entry = {"type": "assistant", "message": {"content": []}}
        assert provider.parse_history_entry(entry) is None


class TestDiscoverTranscript:
    def test_returns_none_or_event(self, provider: AgentProvider) -> None:
        result = provider.discover_transcript("/nonexistent/path", "ccgram:@99")
        # All providers should return None for a nonexistent path
        assert result is None

    def test_accepts_optional_max_age_kwarg(self, provider: AgentProvider) -> None:
        result = provider.discover_transcript(
            "/nonexistent/path",
            "ccgram:@99",
            max_age=0,
        )
        assert result is None


class TestDiscoverCommands:
    def test_returns_list_of_discovered_commands(self, provider: AgentProvider) -> None:
        result = provider.discover_commands("/tmp/nonexistent")
        assert isinstance(result, list)
        assert all(isinstance(c, DiscoveredCommand) for c in result)
        for c in result:
            assert c.name
            assert isinstance(c.description, str)
            assert isinstance(c.source, str)

    def test_builtins_included(self, provider: AgentProvider) -> None:
        result = provider.discover_commands("/tmp/nonexistent")
        names = [c.name for c in result]
        for cmd in provider.capabilities.builtin_commands:
            assert cmd in names
