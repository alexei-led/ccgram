from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ccgram.providers.pi import (
    PiProvider,
    _candidate_transcripts,
    encode_cwd_dirname,
)
from ccgram.providers.pi_format import (
    canonical_tool_name,
    extract_text,
    format_tool_result_text,
    normalize_pending,
    parse_assistant,
    parse_bash_execution,
    parse_session_header,
    parse_tool_result,
    parse_user,
    read_session_header,
)


class TestEncodeCwdDirname:
    @pytest.mark.parametrize(
        ("cwd", "expected"),
        [
            ("/Users/alexei/Workspace/ccgram", "--Users-alexei-Workspace-ccgram--"),
            ("/tmp/foo/", "--tmp-foo--"),
            ("/tmp/foo", "--tmp-foo--"),
            ("/", "----"),
            ("/a", "--a--"),
            ("/a/b", "--a-b--"),
            ("C:\\Users\\x", "--C--Users-x--"),
            ("/has:colon/path", "--has-colon-path--"),
        ],
    )
    def test_encodes(self, cwd: str, expected: str) -> None:
        assert encode_cwd_dirname(cwd) == expected


class TestCanonicalToolName:
    @pytest.mark.parametrize(
        ("raw", "display"),
        [
            ("bash", "Bash"),
            ("BASH", "Bash"),
            ("read", "Read"),
            ("edit", "Edit"),
            ("webfetch", "WebFetch"),
            ("web_fetch", "WebFetch"),
            ("unknown_tool", "unknown_tool"),
        ],
    )
    def test_aliases(self, raw: str, display: str) -> None:
        assert canonical_tool_name(raw) == display


class TestExtractText:
    def test_string(self) -> None:
        assert extract_text("hi") == "hi"

    def test_block_array(self) -> None:
        blocks = [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]
        assert extract_text(blocks) == "a b"

    def test_skips_non_text(self) -> None:
        blocks = [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "visible"},
            {"type": "image", "data": "..."},
        ]
        assert extract_text(blocks) == "visible"

    def test_empty(self) -> None:
        assert extract_text([]) == ""
        assert extract_text(None) == ""
        assert extract_text(123) == ""


class TestFormatToolResultText:
    def test_empty_returns_done(self) -> None:
        assert format_tool_result_text("bash", "") == "Done"

    def test_bash_always_quoted(self) -> None:
        out = format_tool_result_text("bash", "one line")
        assert "1 line" in out
        assert "1 lines" not in out
        assert "one line" in out

    def test_bash_pluralizes_two_lines(self) -> None:
        out = format_tool_result_text("bash", "a\nb")
        assert "2 lines" in out

    def test_short_non_bash_inline(self) -> None:
        assert format_tool_result_text("read", "x") == "x"

    def test_long_non_bash_quoted(self) -> None:
        output = "l1\nl2\nl3\nl4\nl5"
        rendered = format_tool_result_text("read", output)
        assert "5 lines" in rendered
        assert "l1" in rendered


class TestParseSessionHeader:
    def test_ok(self) -> None:
        entry = {
            "type": "session",
            "version": 3,
            "id": "019d9fcf-3663-750a-b941-946136546d38",
            "cwd": "/Users/alexei/Workspace/ccgram",
        }
        assert parse_session_header(entry) == {
            "id": "019d9fcf-3663-750a-b941-946136546d38",
            "cwd": "/Users/alexei/Workspace/ccgram",
        }

    @pytest.mark.parametrize(
        "entry",
        [
            {"type": "message", "id": "x", "cwd": "/"},
            {"type": "session", "cwd": "/"},
            {"type": "session", "id": "x"},
            {"type": "session", "id": "", "cwd": "/"},
            {"type": "session", "id": "x", "cwd": ""},
        ],
    )
    def test_rejects(self, entry: dict) -> None:
        assert parse_session_header(entry) is None


class TestReadSessionHeader:
    def test_reads_first_line(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text(
            '{"type":"session","id":"abc-123","cwd":"/x","version":3}\n'
            '{"type":"message","id":"y"}\n'
        )
        assert read_session_header(str(path)) == {"id": "abc-123", "cwd": "/x"}

    def test_missing_file(self, tmp_path: Path) -> None:
        assert read_session_header(str(tmp_path / "nope.jsonl")) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert read_session_header(str(path)) is None

    def test_malformed_first_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")
        assert read_session_header(str(path)) is None


class TestParseUser:
    def test_text_blocks(self) -> None:
        msg = {"role": "user", "content": [{"type": "text", "text": "analyze g"}]}
        [m] = parse_user(msg)
        assert m.role == "user"
        assert m.content_type == "text"
        assert m.text == "analyze g"

    def test_empty_returns_nothing(self) -> None:
        assert parse_user({"role": "user", "content": []}) == []
        assert parse_user({"role": "user", "content": ""}) == []


class TestParseAssistant:
    def test_text_only(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
        }
        msgs, pending = parse_assistant(msg, {})
        assert len(msgs) == 1
        assert msgs[0].content_type == "text"
        assert msgs[0].text == "hello"
        assert pending == {}

    def test_text_and_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running tools"},
                {
                    "type": "toolCall",
                    "id": "t1",
                    "name": "bash",
                    "arguments": {"command": "ls -la"},
                },
                {"type": "text", "text": "after"},
                {
                    "type": "toolCall",
                    "id": "t2",
                    "name": "read",
                    "arguments": {"path": "foo.py"},
                },
            ],
        }
        msgs, pending = parse_assistant(msg, {}, timestamp="2024-12-03T14:00:02.000Z")
        assert [m.content_type for m in msgs] == [
            "text",
            "tool_use",
            "text",
            "tool_use",
        ]
        assert [m.text for m in msgs] == [
            "running tools",
            "\U0001f4bb **bash**: `ls -la`",
            "after",
            "\U0001f4d6 **read**: `foo.py`",
        ]
        assert all(m.timestamp == "2024-12-03T14:00:02.000Z" for m in msgs)
        assert msgs[1].tool_use_id == "t1"
        assert msgs[1].tool_name == "Bash"
        assert msgs[3].tool_use_id == "t2"
        assert msgs[3].tool_name == "Read"
        assert pending == {"t1": ("bash", "Bash"), "t2": ("read", "Read")}

    def test_skips_thinking(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "public"},
            ],
        }
        msgs, _ = parse_assistant(msg, {})
        assert len(msgs) == 1
        assert msgs[0].text == "public"

    def test_api_error_surfaces(self) -> None:
        msg = {
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": '400 {"error":"bad model"}',
        }
        msgs, _ = parse_assistant(msg, {})
        assert len(msgs) == 1
        assert msgs[0].content_type == "text"
        assert "API error" in msgs[0].text
        assert "bad model" in msgs[0].text

    def test_empty_content_no_error_no_output(self) -> None:
        msgs, _ = parse_assistant({"role": "assistant", "content": []}, {})
        assert msgs == []

    def test_error_appended_alongside_partial_content(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "partial reply "}],
            "stopReason": "error",
            "errorMessage": "rate limited",
        }
        msgs, _ = parse_assistant(msg, {})
        assert [m.content_type for m in msgs] == ["text", "text"]
        assert msgs[0].text == "partial reply"
        assert "API error" in msgs[1].text
        assert "rate limited" in msgs[1].text

    def test_string_content(self) -> None:
        msg = {"role": "assistant", "content": "plain string reply"}
        msgs, _ = parse_assistant(msg, {})
        assert len(msgs) == 1
        assert msgs[0].text == "plain string reply"
        assert msgs[0].content_type == "text"

    def test_string_content_with_error(self) -> None:
        msg = {
            "role": "assistant",
            "content": "partial",
            "stopReason": "error",
            "errorMessage": "context overflow",
        }
        msgs, _ = parse_assistant(msg, {})
        assert len(msgs) == 2
        assert msgs[0].text == "partial"
        assert "API error" in msgs[1].text
        assert "context overflow" in msgs[1].text


class TestParseToolResult:
    def test_pairs_with_pending(self) -> None:
        pending = {"t1": ("bash", "Bash")}
        msg = {
            "role": "toolResult",
            "toolCallId": "t1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "one\ntwo\nthree\nfour"}],
            "isError": False,
        }
        [out], pending = parse_tool_result(msg, pending)
        assert out.content_type == "tool_result"
        assert out.tool_use_id == "t1"
        assert out.tool_name == "Bash"
        assert "4 lines" in out.text
        assert pending == {}

    def test_fallback_without_pending(self) -> None:
        msg = {
            "role": "toolResult",
            "toolCallId": "unknown",
            "toolName": "read",
            "content": [{"type": "text", "text": "ok"}],
        }
        [out], _ = parse_tool_result(msg, {})
        assert out.tool_name == "Read"
        assert out.text == "ok"

    def test_error_flag(self) -> None:
        msg = {
            "role": "toolResult",
            "toolCallId": "t1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "boom"}],
            "isError": True,
        }
        [out], _ = parse_tool_result(msg, {})
        assert out.text == "Error: boom"

    def test_empty_error_is_not_done(self) -> None:
        msg = {
            "role": "toolResult",
            "toolCallId": "t1",
            "toolName": "bash",
            "content": [],
            "isError": True,
        }
        [out], _ = parse_tool_result(msg, {})
        assert out.text == "Error"


class TestParseBashExecution:
    def test_happy_path(self) -> None:
        [out] = parse_bash_execution(
            {"role": "bashExecution", "command": "ls", "output": "a\nb"}
        )
        assert "$ ls" in out.text
        assert "a\nb" in out.text

    def test_excluded_from_context(self) -> None:
        assert (
            parse_bash_execution(
                {
                    "role": "bashExecution",
                    "command": "echo x",
                    "output": "x",
                    "excludeFromContext": True,
                }
            )
            == []
        )

    def test_non_zero_exit(self) -> None:
        [out] = parse_bash_execution(
            {"role": "bashExecution", "command": "false", "output": "", "exitCode": 1},
            timestamp="2024-12-03T14:00:03.000Z",
        )
        assert "exit code 1" in out.text
        assert out.timestamp == "2024-12-03T14:00:03.000Z"


class TestNormalizePending:
    def test_accepts_tuple(self) -> None:
        assert normalize_pending({"x": ("bash", "Bash")}) == {"x": ("bash", "Bash")}

    def test_accepts_legacy_string(self) -> None:
        assert normalize_pending({"x": "bash"}) == {"x": ("bash", "Bash")}

    def test_rejects_garbage(self) -> None:
        assert normalize_pending({"x": 123, "y": None}) == {}

    def test_non_dict(self) -> None:
        assert normalize_pending(None) == {}
        assert normalize_pending([]) == {}


class TestMakeLaunchArgs:
    def setup_method(self) -> None:
        self.provider = PiProvider()

    def test_fresh(self) -> None:
        assert self.provider.make_launch_args() == ""

    def test_continue(self) -> None:
        assert self.provider.make_launch_args(use_continue=True) == "--continue"

    def test_session_by_uuid(self) -> None:
        uuid = "019d9fcf-3663-750a-b941-946136546d38"
        assert self.provider.make_launch_args(resume_id=uuid) == f"--session {uuid}"

    def test_session_by_path(self) -> None:
        args = self.provider.make_launch_args(resume_id="/tmp/a.jsonl")
        assert args == "--session /tmp/a.jsonl"

    def test_shell_quoting_path_with_space(self) -> None:
        args = self.provider.make_launch_args(resume_id="/tmp/has space.jsonl")
        assert args == "--session '/tmp/has space.jsonl'"

    def test_resume_wins_over_continue(self) -> None:
        args = self.provider.make_launch_args(resume_id="x", use_continue=True)
        assert args == "--session x"


class TestParseTranscriptLine:
    def setup_method(self) -> None:
        self.provider = PiProvider()

    def test_passes_through_session_header(self) -> None:
        line = '{"type":"session","id":"abc","cwd":"/x"}'
        out = self.provider.parse_transcript_line(line)
        assert out == {"type": "session", "id": "abc", "cwd": "/x"}

    def test_unwraps_message_envelope(self) -> None:
        line = json.dumps(
            {
                "type": "message",
                "id": "m1",
                "parentId": "m0",
                "timestamp": "2024-12-03T14:00:01.000Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
        out = self.provider.parse_transcript_line(line)
        assert out is not None
        assert out["type"] == "user"
        assert out["id"] == "m1"
        assert out["parentId"] == "m0"
        assert out["timestamp"] == "2024-12-03T14:00:01.000Z"
        assert out["message"]["content"][0]["text"] == "hi"

    def test_rejects_empty(self) -> None:
        assert self.provider.parse_transcript_line("") is None
        assert self.provider.parse_transcript_line("not json") is None

    def test_rejects_message_without_role(self) -> None:
        line = '{"type":"message","message":{"content":[]}}'
        assert self.provider.parse_transcript_line(line) is None


class TestHistoryEntry:
    def test_reads_raw_user_message(self) -> None:
        entry = {
            "type": "message",
            "timestamp": "2024-12-03T14:00:01.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        }
        assert PiProvider().is_user_transcript_entry(entry) is True
        parsed = PiProvider().parse_history_entry(entry)
        assert parsed is not None
        assert parsed.text == "hi"
        assert parsed.timestamp == "2024-12-03T14:00:01.000Z"

    def test_parses_assistant_message(self) -> None:
        entry = {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        assert PiProvider().is_user_transcript_entry(entry) is False
        parsed = PiProvider().parse_history_entry(entry)
        assert parsed is not None
        assert parsed.role == "assistant"
        assert parsed.text == "hi"

    def test_preserves_order_and_timestamp(self) -> None:
        provider = PiProvider()
        entry = {
            "type": "assistant",
            "timestamp": "2024-12-03T14:00:02.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "read",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "text", "text": "after"},
                ]
            },
        }
        msgs, pending = provider.parse_transcript_entries([entry], {})
        assert [m.content_type for m in msgs] == ["text", "tool_use", "text"]
        assert [m.text for m in msgs] == [
            "before",
            "\U0001f4d6 **read**: `foo.py`",
            "after",
        ]
        assert all(m.timestamp == "2024-12-03T14:00:02.000Z" for m in msgs)
        assert pending == {"t1": ("read", "Read")}


class TestDiscoverTranscript:
    def _write_session(self, path: Path, session_id: str, cwd: str) -> Path:
        path.write_text(
            json.dumps({"type": "session", "id": session_id, "cwd": cwd, "version": 3})
            + "\n"
        )
        return path

    def test_returns_newest_matching_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ccgram.providers.pi._pi_sessions_dir", lambda: tmp_path)
        cwd = "/real/project"
        session_dir = tmp_path / encode_cwd_dirname(cwd)
        session_dir.mkdir()
        older = self._write_session(session_dir / "old.jsonl", "s1", cwd)
        newer = self._write_session(session_dir / "new.jsonl", "s2", cwd)
        now = time.time()
        import os

        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))

        monkeypatch.setattr("pathlib.Path.resolve", lambda self, strict=False: self)
        provider = PiProvider()
        ev = provider.discover_transcript(cwd, "ccgram:@0", max_age=0)
        assert ev is not None
        assert ev.session_id == "s2"
        assert ev.transcript_path == str(newer)
        assert ev.window_key == "ccgram:@0"

    def test_empty_cwd_returns_none(self) -> None:
        assert PiProvider().discover_transcript("", "ccgram:@0") is None

    def test_missing_dir_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ccgram.providers.pi._pi_sessions_dir", lambda: tmp_path)
        assert PiProvider().discover_transcript("/no/such/place", "ccgram:@0") is None

    def test_rejects_stale_files_when_max_age_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ccgram.providers.pi._pi_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr("pathlib.Path.resolve", lambda self, strict=False: self)
        cwd = "/some/project"
        session_dir = tmp_path / encode_cwd_dirname(cwd)
        session_dir.mkdir()
        stale = self._write_session(session_dir / "old.jsonl", "s1", cwd)
        import os

        now = time.time()
        os.utime(stale, (now - 600, now - 600))

        provider = PiProvider()
        assert provider.discover_transcript(cwd, "ccgram:@0", max_age=60) is None
        ev = provider.discover_transcript(cwd, "ccgram:@0", max_age=1200)
        assert ev is not None and ev.session_id == "s1"


class TestCapabilities:
    def test_shape(self) -> None:
        caps = PiProvider().capabilities
        assert caps.name == "pi"
        assert caps.launch_command == "pi"
        assert caps.supports_hook is True
        assert caps.supports_hook_events is True
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.transcript_format == "jsonl"
        assert caps.supports_incremental_read is True
        assert set(caps.builtin_commands) == {
            "/changelog",
            "/clear",
            "/clone",
            "/colors",
            "/compact",
            "/copy",
            "/debug",
            "/export",
            "/fork",
            "/hotkeys",
            "/import",
            "/login",
            "/logout",
            "/model",
            "/name",
            "/quit",
            "/reload",
            "/session",
            "/settings",
            "/share",
            "/tree",
        }
        assert "/new" not in caps.builtin_commands
        assert "/resume" not in caps.builtin_commands

        assert caps.tui_picker_commands == frozenset(
            {"model", "login", "fork", "clone", "import", "settings"}
        )


class TestDiscoverCommands:
    def test_returns_builtins_and_dynamic_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        pi_agent = home / ".pi" / "agent"
        skill_dir = pi_agent / "skills" / "brave-search"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: brave-search\ndescription: Search the web\nuser-invocable: true\n---\n"
        )
        prompt_dir = pi_agent / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "review.md").write_text(
            "---\ndescription: Review staged changes\nargument-hint: <PR>\n---\n"
        )
        ext_dir = pi_agent / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "stats.ts").write_text(
            'export default function (pi) { pi.registerCommand("stats", { description: "Stats", handler: async () => {} }); }\n'
        )
        monkeypatch.setattr(Path, "home", lambda: home)

        cmds = PiProvider().discover_commands(str(tmp_path / "proj"))
        names = {c.name for c in cmds}
        assert "/clear" in names
        assert "/tree" in names
        assert "/model" in names
        assert "/quit" in names
        assert "brave-search" in names
        assert "review" in names
        assert "stats" in names
        assert all(c.name for c in cmds)

    def test_dedup_keeps_first_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        pi_agent = home / ".pi" / "agent"
        skill_dir = pi_agent / "skills" / "stats"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: stats\ndescription: Skill stats\nuser-invocable: true\n---\n"
        )
        prompt_dir = pi_agent / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "stats.md").write_text("---\ndescription: Prompt stats\n---\n")
        ext_dir = pi_agent / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "stats.ts").write_text(
            'export default function (pi) { pi.registerCommand("stats", { description: "Extension stats", handler: async () => {} }); }\n'
        )
        monkeypatch.setattr(Path, "home", lambda: home)

        cmds = PiProvider().discover_commands(str(tmp_path / "proj"))
        stats = [cmd for cmd in cmds if cmd.name == "stats"]
        assert len(stats) == 1
        assert stats[0].source == "skill"
        assert stats[0].description == "Skill stats"

    def test_ignores_false_positive_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        pi_agent = home / ".pi" / "agent"
        skill_dir = pi_agent / "skills" / "broken"
        skill_dir.mkdir(parents=True)
        (skill_dir / "README.md").write_text("just docs\n")
        prompt_dir = pi_agent / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "skip.txt").write_text("---\ndescription: ignored\n---\n")
        ext_dir = pi_agent / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "noop.ts").write_text("export default function () { return; }\n")
        monkeypatch.setattr(Path, "home", lambda: home)

        cmds = PiProvider().discover_commands(str(tmp_path / "proj"))
        names = {cmd.name for cmd in cmds}
        assert "broken" not in names
        assert "skip" not in names
        assert "noop" not in names


class TestIntegrationWithCandidateTranscripts:
    def test_sorts_newest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ccgram.providers.pi._pi_sessions_dir", lambda: tmp_path)
        cwd = "/demo"
        d = tmp_path / encode_cwd_dirname(cwd)
        d.mkdir()
        (d / "a.jsonl").write_text("{}\n")
        (d / "b.jsonl").write_text("{}\n")
        import os

        now = time.time()
        os.utime(d / "a.jsonl", (now - 200, now - 200))
        os.utime(d / "b.jsonl", (now, now))
        result = _candidate_transcripts(cwd)
        assert [p.name for _, p in result] == ["b.jsonl", "a.jsonl"]
