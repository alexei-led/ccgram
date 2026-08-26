import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.providers.claude import (
    ClaudeProvider,
    _find_mode_line,
    _mode_short_label,
)
from ccgram.providers._resume import discover_claude_sessions, index_message_count
from ccgram.providers.codex import CodexProvider
from ccgram.providers.gemini import GeminiProvider
from ccgram.providers.shell import ShellProvider


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        pytest.param(ClaudeProvider, True, id="claude"),
        pytest.param(CodexProvider, False, id="codex"),
        pytest.param(GeminiProvider, False, id="gemini"),
        pytest.param(ShellProvider, False, id="shell"),
    ],
)
def test_only_claude_has_yolo_confirmation(cls, expected) -> None:
    assert cls().capabilities.has_yolo_confirmation is expected


_PICKER_COMMANDS = {
    # "status" is deliberately absent from Claude's set: it prints a report
    # rather than opening a TUI picker.
    ClaudeProvider: {
        "agents",
        "copy",
        "diff",
        "effort",
        "model",
        "permissions",
        "release-notes",
        "rewind",
        "settings",
        "skills",
        "theme",
        "tui",
    },
    CodexProvider: {"model", "permissions", "skills", "statusline", "personality"},
    GeminiProvider: {
        "agents",
        "auth",
        "chat",
        "editor",
        "extensions",
        "ide",
        "model",
        "privacy",
        "rewind",
        "settings",
        "terminal-setup",
        "theme",
    },
}


@pytest.mark.parametrize(
    ("cls", "expected"),
    list(_PICKER_COMMANDS.items()),
    ids=[cls().capabilities.name for cls in _PICKER_COMMANDS],
)
def test_picker_commands_exact_set(cls, expected) -> None:
    assert cls().capabilities.tui_picker_commands == frozenset(expected)


class TestScrapeCurrentMode:
    @staticmethod
    def _capture(**kwargs):
        return patch("ccgram.multiplexer.multiplexer", capture_pane=AsyncMock(**kwargs))

    @pytest.mark.parametrize(
        ("pane", "expected"),
        [
            pytest.param("some output\n⏵⏵ auto-accept edits on  >", "Edit", id="edit"),
            pytest.param("some output\n⏸ plan mode  >", "Plan", id="plan"),
            pytest.param("some output\n⏵⏵ bypass permissions  >", "YOLO", id="yolo"),
            pytest.param("just regular output\nno mode here", None, id="no_mode_line"),
            pytest.param("", None, id="empty_pane"),
        ],
    )
    async def test_maps_pane_chrome_to_mode_label(self, pane, expected):
        with self._capture(return_value=pane):
            result = await ClaudeProvider().scrape_current_mode("@0")
        assert result == expected

    async def test_capture_failure_returns_none(self):
        with self._capture(side_effect=OSError("tmux gone")):
            assert await ClaudeProvider().scrape_current_mode("@0") is None

    async def test_shell_provider_has_no_mode(self):
        assert await ShellProvider().scrape_current_mode("@0") is None


class TestFindModeLine:
    def test_finds_chrome_marker(self):
        pane = "output\n─────\n⏵⏵ auto-accept edits on  >"
        result = _find_mode_line(pane)
        assert result is not None
        assert "auto-accept" in result

    def test_returns_none_for_no_mode(self):
        assert _find_mode_line("just some text\nno markers") is None

    def test_hint_fallback(self):
        pane = "line1\nline2\nauto mode enabled\nlast"
        result = _find_mode_line(pane)
        assert result is not None
        assert "auto mode" in result


class TestModeShortLabel:
    @pytest.mark.parametrize(
        ("mode_line", "expected"),
        [
            ("⏵⏵ auto-accept edits on  >", "Edit"),
            ("⏸ plan mode  >", "Plan"),
            ("⏵⏵ bypass permissions  >", "YOLO"),
            ("⏵⏵ auto mode  >", "Auto"),
        ],
    )
    def test_known_labels(self, mode_line, expected):
        assert _mode_short_label(mode_line) == expected

    def test_unknown_returns_none(self):
        assert _mode_short_label("something weird") is None


class TestParseTranscriptEntries:
    """Characterization: parse_transcript_entries wraps ParsedEntry fields into AgentMessage."""

    def _entry(self, msg_type: str, content: list) -> dict:
        return {
            "type": msg_type,
            "message": {"content": content},
            "timestamp": "2024-01-01T00:00:00.000Z",
        }

    def test_assistant_text_wrapped(self):
        provider = ClaudeProvider()
        entries = [self._entry("assistant", [{"type": "text", "text": "hello world"}])]
        messages, remaining = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        m = messages[0]
        assert m.role == "assistant"
        assert m.content_type == "text"
        assert m.text == "hello world"
        assert m.timestamp == "2024-01-01T00:00:00.000Z"
        assert not remaining

    def test_tool_use_and_result_wrapped(self):
        provider = ClaudeProvider()
        entries = [
            self._entry(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "x.py"},
                    }
                ],
            ),
            self._entry(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "3 lines\nof\ntext",
                    }
                ],
            ),
        ]
        messages, remaining = provider.parse_transcript_entries(entries, {})
        tool_use_msgs = [m for m in messages if m.content_type == "tool_use"]
        tool_result_msgs = [m for m in messages if m.content_type == "tool_result"]
        assert len(tool_use_msgs) == 1
        assert tool_use_msgs[0].tool_use_id == "t1"
        assert tool_use_msgs[0].tool_name == "Read"
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].tool_use_id == "t1"
        assert not remaining

    def test_carry_over_pending_tools(self):
        provider = ClaudeProvider()
        entries = [
            self._entry(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            ),
        ]
        messages, remaining = provider.parse_transcript_entries(entries, {})
        assert "t2" in remaining

    def test_unknown_entry_type_skipped(self):
        provider = ClaudeProvider()
        entries = [
            {"type": "summary", "message": {"content": "ignored"}},
            self._entry("assistant", [{"type": "text", "text": "kept"}]),
        ]
        messages, remaining = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "kept"


class TestDiscoverClaudeSessions:
    """``ClaudeProvider.discover_resumable_sessions`` backend — the resume picker
    reads exactly what this returns, so cwd scoping and de-dup are load-bearing."""

    @staticmethod
    def _project(projects: Path, name: str) -> Path:
        project = projects / name
        project.mkdir(parents=True)
        return project

    @staticmethod
    def _index(project: Path, original_path: str, entries: list[dict]) -> None:
        (project / "sessions-index.json").write_text(
            json.dumps({"originalPath": original_path, "entries": entries})
        )

    @staticmethod
    def _transcript(project: Path, session_id: str, cwd: str, prompt: str = "") -> Path:
        path = project / f"{session_id}.jsonl"
        lines = [json.dumps({"cwd": cwd})]
        if prompt:
            lines.append(
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": prompt}}
                )
            )
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_reads_sessions_from_the_index(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        transcript = self._transcript(project, "sess-1", str(tmp_path / "repo"))
        self._index(
            project,
            str(tmp_path / "repo"),
            [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(transcript),
                    "summary": "fix the parser",
                    "messageCount": 7,
                }
            ],
        )

        sessions = discover_claude_sessions(projects)

        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-1"
        assert sessions[0].summary == "fix the parser"
        assert sessions[0].cwd == str((tmp_path / "repo").resolve())
        assert sessions[0].provider_name == "claude"
        assert sessions[0].msg_count == 7

    def test_falls_back_to_bare_jsonl_when_not_indexed(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        self._transcript(project, "sess-bare", str(tmp_path / "repo"), "hello there")

        sessions = discover_claude_sessions(projects)

        assert [s.session_id for s in sessions] == ["sess-bare"]
        assert sessions[0].summary == "hello there"

    def test_index_entry_wins_over_the_same_bare_jsonl(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        transcript = self._transcript(
            project, "dup", str(tmp_path / "repo"), "raw text"
        )
        self._index(
            project,
            str(tmp_path / "repo"),
            [{"sessionId": "dup", "fullPath": str(transcript), "summary": "indexed"}],
        )

        sessions = discover_claude_sessions(projects)

        assert [s.summary for s in sessions] == ["indexed"]

    def test_cwd_filter_matches_the_exact_workspace_only(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        workspaces = {
            "target": tmp_path / "repo",
            "sibling-prefix": tmp_path / "repo-sibling",
            "descendant": tmp_path / "repo" / "nested",
        }
        for session_id, workspace in workspaces.items():
            project = self._project(projects, session_id)
            self._transcript(project, session_id, str(workspace))

        sessions = discover_claude_sessions(projects, cwd=str(tmp_path / "repo"))

        assert [s.session_id for s in sessions] == ["target"]

    def test_sorts_newest_first_and_honours_limit(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        for i, mtime in enumerate((100.0, 300.0, 200.0)):
            path = self._transcript(project, f"s{i}", str(tmp_path / "repo"))
            os.utime(path, (mtime, mtime))

        assert [s.session_id for s in discover_claude_sessions(projects)] == [
            "s1",
            "s2",
            "s0",
        ]
        assert [s.session_id for s in discover_claude_sessions(projects, limit=2)] == [
            "s1",
            "s2",
        ]

    def test_summary_falls_back_to_first_prompt_then_session_id(
        self, tmp_path: Path
    ) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        t1 = self._transcript(project, "prompt-only-session", str(tmp_path / "repo"))
        t2 = self._transcript(project, "no-summary-session-id", str(tmp_path / "repo"))
        self._index(
            project,
            str(tmp_path / "repo"),
            [
                {
                    "sessionId": "prompt-only-session",
                    "fullPath": str(t1),
                    "firstPrompt": "from firstPrompt",
                },
                {"sessionId": "no-summary-session-id", "fullPath": str(t2)},
            ],
        )

        by_id = {s.session_id: s.summary for s in discover_claude_sessions(projects)}

        assert by_id["prompt-only-session"] == "from firstPrompt"
        assert by_id["no-summary-session-id"] == "no-summary-se"[:12]

    @pytest.mark.parametrize(
        "count_key", ["messageCount", "msgCount", "msg_count", "messages"]
    )
    def test_message_count_accepts_every_index_spelling(
        self, tmp_path: Path, count_key: str
    ) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        transcript = self._transcript(project, "s", str(tmp_path / "repo"))
        self._index(
            project,
            str(tmp_path / "repo"),
            [{"sessionId": "s", "fullPath": str(transcript), count_key: 12}],
        )

        assert discover_claude_sessions(projects)[0].msg_count == 12

    @pytest.mark.parametrize(
        "value", [0, -1, "12", None], ids=["zero", "negative", "string", "missing"]
    )
    def test_non_positive_message_count_is_dropped(self, value: object) -> None:
        assert index_message_count({"messageCount": value}) is None

    def test_missing_projects_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_claude_sessions(tmp_path / "nope") == []

    def test_unresolvable_cwd_returns_empty(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        self._transcript(project, "s", str(tmp_path / "repo"))

        assert discover_claude_sessions(projects, cwd="\x00bad") == []

    @pytest.mark.parametrize(
        "index_text",
        [
            pytest.param("{not json", id="corrupt_json"),
            pytest.param('["not", "a", "dict"]', id="top_level_list"),
            pytest.param('{"entries": "not-a-list"}', id="entries_not_a_list"),
            pytest.param('{"entries": ["not-a-dict"]}', id="entry_not_a_dict"),
        ],
    )
    def test_unusable_index_still_yields_the_bare_jsonl_scan(
        self, tmp_path: Path, index_text: str
    ) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        self._transcript(project, "sess-bare", str(tmp_path / "repo"))
        (project / "sessions-index.json").write_text(index_text)

        assert [s.session_id for s in discover_claude_sessions(projects)] == [
            "sess-bare"
        ]

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({"fullPath": "/x.jsonl"}, id="no_session_id"),
            pytest.param(
                {"sessionId": "", "fullPath": "/x.jsonl"}, id="empty_session_id"
            ),
            pytest.param({"sessionId": "s"}, id="no_full_path"),
            pytest.param({"sessionId": "s", "fullPath": ""}, id="empty_full_path"),
            pytest.param(
                {"sessionId": "s", "fullPath": "/x.jsonl", "projectPath": 42},
                id="non_string_project_path",
            ),
            pytest.param(
                {"sessionId": "s", "fullPath": "/does/not/exist.jsonl"},
                id="transcript_file_gone",
            ),
        ],
    )
    def test_unusable_index_entries_are_skipped(
        self, tmp_path: Path, entry: dict
    ) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        self._index(project, str(tmp_path / "repo"), [entry])

        assert discover_claude_sessions(projects) == []

    def test_bare_jsonl_without_cwd_is_skipped(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        project = self._project(projects, "repo")
        (project / "no-cwd.jsonl").write_text(json.dumps({"type": "user"}) + "\n")

        assert discover_claude_sessions(projects) == []

    def test_files_directly_under_projects_are_ignored(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        projects.mkdir()
        (projects / "stray.jsonl").write_text(json.dumps({"cwd": str(tmp_path)}) + "\n")

        assert discover_claude_sessions(projects) == []
