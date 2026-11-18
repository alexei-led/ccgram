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
from ccgram.providers.pi_format import read_session_header
from ccgram.session_protocol.messages import (
    SourceEntry,
    build_messages,
    parse_records,
    read_entries,
)
from ccgram.transcript_reader import render_protocol_records


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


class TestMakeLaunchArgs:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({}, "", id="fresh"),
            pytest.param({"use_continue": True}, "--continue", id="continue"),
            pytest.param(
                {"resume_id": "019d9fcf-3663-750a-b941-946136546d38"},
                "--session 019d9fcf-3663-750a-b941-946136546d38",
                id="session_by_uuid",
            ),
            pytest.param(
                {"resume_id": "/tmp/a.jsonl"},
                "--session /tmp/a.jsonl",
                id="session_by_path",
            ),
            pytest.param(
                {"resume_id": "/tmp/has space.jsonl"},
                "--session '/tmp/has space.jsonl'",
                id="path_with_space_is_shell_quoted",
            ),
            pytest.param(
                {"resume_id": "x", "use_continue": True},
                "--session x",
                id="resume_wins_over_continue",
            ),
        ],
    )
    def test_make_launch_args(self, kwargs: dict, expected: str) -> None:
        assert PiProvider().make_launch_args(**kwargs) == expected


class TestCanonicalTranscriptParsing:
    def test_reads_pi_message_without_provider_rewriting(self) -> None:
        raw = (
            json.dumps(
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
            ).encode()
            + b"\n"
        )
        entries = read_entries(raw, "pi")
        messages = build_messages(entries, "pi-test", "pi")
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].time == "2024-12-03T14:00:01.000Z"
        assert messages[0].parts[0].text == "hi"

    def test_ignores_non_message_and_malformed_records(self) -> None:
        entries = read_entries(b'{"type":"session","id":"abc"}\nnot json\n', "pi")
        records, _ = parse_records(entries, "pi")
        assert records == ()
        assert read_entries(b"not json\n", "pi") == []

    @pytest.mark.parametrize(
        ("payload", "expected_text", "expected_level"),
        [
            pytest.param(
                {"command": "ls", "output": "a\nb"},
                "$ ls\na\nb",
                "info",
                id="success",
            ),
            pytest.param(
                {"command": "false", "exitCode": 2},
                "$ false\nexit code 2",
                "error",
                id="non-zero-exit",
            ),
            pytest.param(
                {"command": "sleep 10", "output": "stopped", "cancelled": True},
                "$ sleep 10\nstopped\n(cancelled)",
                "info",
                id="cancelled",
            ),
        ],
    )
    @pytest.mark.parametrize("envelope_type", ["bashExecution", "message"])
    def test_bash_execution_is_visible_in_history_and_telegram_rendering(
        self,
        payload: dict,
        expected_text: str,
        expected_level: str,
        envelope_type: str,
    ) -> None:
        message = {
            "role": "bashExecution",
            "timestamp": "2024-12-03T14:00:03.000Z",
            **payload,
        }
        entry = SourceEntry(
            0,
            {"type": envelope_type, "id": "bash-1", "message": message},
        )

        records, state = parse_records([entry], "pi")
        messages = build_messages([entry], "pi-test", "pi")
        rendered = render_protocol_records("pi-test", "pi", records, state)

        assert len(messages) == 1
        assert messages[0].role == "agent"
        assert messages[0].time == "2024-12-03T14:00:03.000Z"
        assert messages[0].parts[0].type == "notice"
        assert messages[0].parts[0].text == expected_text
        assert messages[0].parts[0].level == expected_level
        assert len(rendered) == 1
        assert rendered[0].text == expected_text
        assert rendered[0].role == "assistant"

    def test_bash_execution_excluded_from_history_and_telegram(self) -> None:
        entry = SourceEntry(
            0,
            {
                "type": "bashExecution",
                "id": "excluded-bash",
                "message": {
                    "role": "bashExecution",
                    "command": "echo secret",
                    "output": "secret",
                    "excludeFromContext": True,
                },
            },
        )

        records, state = parse_records([entry], "pi")

        assert records == ()
        assert build_messages([entry], "pi-test", "pi") == ()
        assert render_protocol_records("pi-test", "pi", records, state) == []

    def test_history_message_preserves_order_and_timestamp(self) -> None:
        entry = {
            "type": "message",
            "timestamp": "2024-12-03T14:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "read",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "text", "text": "after"},
                ],
            },
        }
        messages = build_messages([SourceEntry(0, entry)], "pi-test", "pi")
        assert len(messages) == 1
        assert [part.type for part in messages[0].parts] == ["text", "tool", "text"]
        assert [part.text for part in messages[0].parts] == ["before", None, "after"]
        assert messages[0].time == "2024-12-03T14:00:02.000Z"

    @pytest.mark.parametrize(
        ("content", "expected_texts"),
        [
            pytest.param([], ["Provider error"], id="empty-content"),
            pytest.param(
                [{"type": "text", "text": "partial output"}],
                ["partial output", "Provider error"],
                id="partial-content",
            ),
        ],
    )
    def test_assistant_error_is_preserved_with_empty_or_partial_content(
        self, content: list[dict], expected_texts: list[str]
    ) -> None:
        entry = {
            "type": "message",
            "id": "error-message",
            "message": {
                "role": "assistant",
                "content": content,
                "stopReason": "error",
                "errorMessage": "Provider error",
            },
        }

        messages = build_messages([SourceEntry(0, entry)], "pi-test", "pi")

        assert [part.text for part in messages[0].parts] == expected_texts
        assert messages[0].parts[-1].level == "error"


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
        assert caps.hook_install_managed_by_ccgram is True
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_incremental_read is True
        assert set(caps.builtin_commands) == {
            "/changelog",
            "/clone",
            "/compact",
            "/copy",
            "/export",
            "/followup",
            "/fork",
            "/hotkeys",
            "/login",
            "/logout",
            "/model",
            "/name",
            "/new",
            "/quit",
            "/reload",
            "/scoped-models",
            "/session",
            "/settings",
            "/share",
            "/thinking",
            "/tree",
        }
        assert "/clear" not in caps.builtin_commands
        assert "/resume" not in caps.builtin_commands

        assert caps.tui_picker_commands == frozenset(
            {
                "model",
                "thinking",
                "login",
                "logout",
                "fork",
                "clone",
                "scoped-models",
                "settings",
                "tree",
            }
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
        assert "/new" in names
        assert "/clear" not in names
        assert "/followup" in names
        assert "/scoped-models" in names
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
