import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from ccgram.providers.antigravity import AntigravityProvider
from ccgram.providers.base import (
    AgentMessage,
    AgentProvider,
    DiscoveredCommand,
    ProviderCapabilities,
    StatusUpdate,
)
from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.providers.gemini import GeminiProvider
from ccgram.providers.opencode import OpenCodeProvider
from ccgram.providers.pi import PiProvider
from ccgram.providers.shell import ShellProvider


class StubProvider(JsonlProvider):
    _CAPS = ProviderCapabilities(
        name="stub",
        launch_command="stub-cli",
        supports_hook=True,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
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


PROVIDER_FIXTURES: list[type] = [
    StubProvider,
    AntigravityProvider,
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    OpenCodeProvider,
    PiProvider,
    ShellProvider,
]


@pytest.fixture(params=PROVIDER_FIXTURES, ids=lambda cls: cls.__name__)
def provider(request: pytest.FixtureRequest) -> AgentProvider:
    return request.param()


class TestAgentProviderCapabilities:
    def test_required_fields(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        assert caps.name
        if caps.name != "shell":
            assert caps.launch_command

    def test_immutability(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        with pytest.raises(FrozenInstanceError):
            caps.name = "hacked"  # type: ignore[misc]

    def test_resume_picker_requires_resume_support(
        self, provider: AgentProvider
    ) -> None:
        caps = provider.capabilities
        assert not caps.supports_resume_picker or caps.supports_resume
        assert callable(provider.discover_resumable_sessions)

    def test_only_claude_uses_pyte_status_parsing(
        self, provider: AgentProvider
    ) -> None:
        caps = provider.capabilities
        assert caps.uses_pyte_status_parsing == (caps.name == "claude")


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


class TestParseTranscriptLine:
    @pytest.mark.parametrize(
        "line",
        ["", "   ", "not json at all"],
        ids=["empty", "whitespace", "invalid"],
    )
    def test_invalid_returns_none(self, provider: AgentProvider, line: str) -> None:
        assert provider.parse_transcript_line(line) is None

    def test_valid_returns_dict(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        line = json.dumps({"type": "assistant", "message": {"content": "hi"}})
        result = provider.parse_transcript_line(line)
        assert isinstance(result, dict)
        assert result["type"] == "assistant"


def _make_assistant_entry(
    provider: AgentProvider, text: str = "hello"
) -> dict[str, Any]:
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
    if name == "antigravity":
        return {"type": "PLANNER_RESPONSE", "source": "MODEL", "content": text}
    if name == "opencode":
        return {
            "type": "opencode_part",
            "role": "assistant",
            "part": {"type": "text", "text": text},
        }
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _make_tool_use_entry(provider: AgentProvider) -> dict[str, Any]:
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
    if name == "antigravity":
        return {
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "tool_calls": [{"id": "t1", "name": "Read"}],
        }
    if name == "opencode":
        return {
            "type": "opencode_part",
            "role": "assistant",
            "part": {
                "type": "tool",
                "callID": "t1",
                "tool": "bash",
                "state": {"status": "running", "input": {"command": "ls"}},
            },
        }
    if name == "pi":
        return {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "read",
                        "arguments": {"path": "foo.py"},
                    }
                ]
            },
        }
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]
        },
    }


def _make_tool_result_entry(provider: AgentProvider) -> dict[str, Any]:
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
        return {"type": "gemini", "content": "result ok"}
    if name == "antigravity":
        return {
            "type": "TOOL_RESULT",
            "source": "MODEL",
            "tool_call_id": "t1",
            "content": "ok",
        }
    if name == "opencode":
        return {
            "type": "opencode_part",
            "role": "assistant",
            "part": {
                "type": "tool",
                "callID": "t1",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "output": "ok",
                },
            },
        }
    if name == "pi":
        return {
            "type": "toolResult",
            "message": {
                "role": "toolResult",
                "toolCallId": "t1",
                "toolName": "read",
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            },
        }
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
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        entries = [_make_assistant_entry(provider, "hello")]
        messages, _ = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, AgentMessage)
        assert msg.text == "hello"
        assert msg.role == "assistant"

    def test_pending_carry_over(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        entries = [_make_tool_use_entry(provider)]
        _, pending = provider.parse_transcript_entries(entries, {})
        assert "t1" in pending

    def test_pending_resolved_on_result(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        entries = [
            _make_tool_use_entry(provider),
            _make_tool_result_entry(provider),
        ]
        _, pending = provider.parse_transcript_entries(entries, {})
        if provider.capabilities.name == "gemini":
            assert "t1" in pending
        else:
            assert "t1" not in pending


class TestParseTerminalStatus:
    def test_empty_returns_none(self, provider: AgentProvider) -> None:
        assert provider.parse_terminal_status("", pane_title="") is None

    def test_status_update_fields(self, provider: AgentProvider) -> None:
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
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        pane = "some text\n! ls -la\ntotal 42\n"
        result = provider.extract_bash_output(pane, "ls")
        assert result is not None
        assert result.startswith("! ls")


class TestIsUserTranscriptEntry:
    def test_user_entry_detected(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        name = provider.capabilities.name
        if name == "codex":
            entry = {"type": "input_item", "payload": {"role": "user"}}
        elif name == "gemini":
            entry = {"type": "user"}
        elif name == "opencode":
            entry = {
                "type": "opencode_part",
                "role": "user",
                "part": {"type": "text", "text": "hi"},
            }
        else:
            entry = {"type": "user"}
        assert provider.is_user_transcript_entry(entry) is True

    def test_non_user_not_detected(self, provider: AgentProvider) -> None:
        name = provider.capabilities.name
        if name == "codex":
            entry = {"type": "response_item", "payload": {"role": "assistant"}}
        elif name == "gemini":
            entry = {"type": "gemini"}
        elif name == "opencode":
            entry = {
                "type": "opencode_part",
                "role": "assistant",
                "part": {"type": "text", "text": "hi"},
            }
        else:
            entry = {"type": "assistant"}
        assert provider.is_user_transcript_entry(entry) is False

    def test_empty_not_detected(self, provider: AgentProvider) -> None:
        assert provider.is_user_transcript_entry({}) is False


class TestParseHistoryEntry:
    def test_non_message_returns_none(self, provider: AgentProvider) -> None:
        assert provider.parse_history_entry({"type": "summary"}) is None

    def test_assistant_message_parsed(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        entry = _make_assistant_entry(provider, "hello world")
        result = provider.parse_history_entry(entry)
        assert result is not None
        assert isinstance(result, AgentMessage)
        assert result.role == "assistant"
        assert result.text == "hello world"

    def test_user_message_parsed(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_structured_transcript:
            pytest.skip("No transcript support")
        name = provider.capabilities.name
        if name == "codex":
            entry = {
                "type": "input_item",
                "payload": {"role": "user", "content": "my question"},
            }
        elif name == "gemini":
            entry = {"type": "user", "content": "my question"}
        elif name == "antigravity":
            entry = {
                "type": "USER_INPUT",
                "source": "USER_EXPLICIT",
                "content": "my question",
            }
        elif name == "opencode":
            entry = {
                "type": "opencode_part",
                "role": "user",
                "part": {"type": "text", "text": "my question"},
            }
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
        elif name == "antigravity":
            entry = {"type": "PLANNER_RESPONSE", "source": "MODEL", "content": ""}
        elif name == "opencode":
            entry = {
                "type": "opencode_part",
                "role": "assistant",
                "part": {"type": "text", "text": ""},
            }
        else:
            entry = {"type": "assistant", "message": {"content": []}}
        assert provider.parse_history_entry(entry) is None


class TestDiscoverTranscript:
    def test_returns_none_or_event(self, provider: AgentProvider) -> None:
        result = provider.discover_transcript("/nonexistent/path", "ccgram:@99")
        assert result is None

    def test_accepts_optional_max_age_kwarg(self, provider: AgentProvider) -> None:
        result = provider.discover_transcript(
            "/nonexistent/path",
            "ccgram:@99",
            max_age=0,
        )
        assert result is None


class TestStatusSnapshot:
    def test_non_snapshot_providers_return_none(self, provider: AgentProvider) -> None:
        if provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider supports snapshots")
        result = provider.build_status_snapshot(
            "/tmp/nonexistent.jsonl",
            display_name="test",
        )
        assert result is None

    def test_non_snapshot_providers_has_output_false(
        self, provider: AgentProvider
    ) -> None:
        if provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider supports snapshots")
        assert provider.has_output_since("/tmp/nonexistent.jsonl", 0) is False

    def test_snapshot_with_transcript(
        self, provider: AgentProvider, tmp_path: Any
    ) -> None:
        if not provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider does not support snapshots")
        transcript = tmp_path / "codex.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "timestamp": "2026-03-02T17:00:00.000Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "sess-test",
                        "cwd": "/tmp/repo",
                        "cli_version": "0.106.0",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = provider.build_status_snapshot(
            str(transcript),
            display_name="repo",
            session_id="sess-test",
            cwd="/tmp/repo",
        )
        assert result is not None
        assert "repo" in result
        assert "sess-tes" in result

    def test_has_output_since(self, provider: AgentProvider, tmp_path: Any) -> None:
        if not provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider does not support snapshots")
        transcript = tmp_path / "codex.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "timestamp": "2026-03-02T17:00:01.000Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert provider.has_output_since(str(transcript), 0) is True

    def test_has_output_since_missing_file(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider does not support snapshots")
        assert provider.has_output_since("/tmp/nonexistent.jsonl", 0) is False

    def test_snapshot_missing_file_is_safe(self, provider: AgentProvider) -> None:
        if not provider.capabilities.supports_status_snapshot:
            pytest.skip("Provider does not support snapshots")
        result = provider.build_status_snapshot(
            "/tmp/nonexistent.jsonl",
            display_name="test",
        )
        assert result is None or "test" in result

    def test_supports_status_snapshot_flag(self, provider: AgentProvider) -> None:
        caps = provider.capabilities
        if caps.name in ("antigravity", "codex", "gemini", "opencode"):
            assert caps.supports_status_snapshot is True
        else:
            assert caps.supports_status_snapshot is False


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
