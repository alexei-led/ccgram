"""Kimi Code provider tests.

Pane fixtures are verbatim ``tmux capture-pane`` output from kimi 0.31.1
(box borders trimmed to keep lines readable) so the terminal-chrome parsers
are pinned against the real TUI rather than an idealised transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccgram.providers import (
    detect_provider_from_command,
    detect_provider_from_transcript_path,
    resolve_launch_command,
)
from ccgram.providers.kimi import (
    KimiProvider,
    parse_status_modes,
    session_index_path,
    transcript_for_session_dir,
)
from ccgram.providers.kimi_format import (
    epoch_ms_to_iso,
    extract_text,
    format_tool_result_text,
    normalize_pending,
    parse_content_part,
    parse_tool_call,
    parse_tool_result,
    parse_turn_prompt,
    tool_call_summary,
)
from ccgram.providers.process_detection import classify_provider_from_argv

RULE = "─" * 118


@pytest.fixture
def kimi() -> KimiProvider:
    return KimiProvider()


# ── Capabilities & launch ────────────────────────────────────────────────


class TestCapabilities:
    def test_identity(self, kimi: KimiProvider) -> None:
        caps = kimi.capabilities
        assert caps.name == "kimi"
        assert caps.launch_command == "kimi"

    def test_is_hookless_with_resume(self, kimi: KimiProvider) -> None:
        caps = kimi.capabilities
        assert caps.supports_hook is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_incremental_read is True

    def test_yolo_flag_needs_no_tui_confirmation(self, kimi: KimiProvider) -> None:
        # ``kimi --yolo`` enters permissive mode from the flag alone.
        assert kimi.capabilities.has_yolo_confirmation is False

    def test_picker_commands_are_builtins(self, kimi: KimiProvider) -> None:
        caps = kimi.capabilities
        builtins = {name.lstrip("/") for name in caps.builtin_commands}
        assert caps.tui_picker_commands <= builtins

    def test_sessions_command_not_exposed(self, kimi: KimiProvider) -> None:
        # /sessions collides with ccgram's own session picker.
        assert "/sessions" not in kimi.capabilities.builtin_commands


class TestLaunchArgs:
    def test_fresh_session(self, kimi: KimiProvider) -> None:
        assert kimi.make_launch_args() == ""

    def test_resume_uses_short_flag(self, kimi: KimiProvider) -> None:
        session = "session_24dbec47-d2ee-410f-9b08-3d64973cc015"
        assert kimi.make_launch_args(resume_id=session) == f"-S {session}"

    def test_continue(self, kimi: KimiProvider) -> None:
        assert kimi.make_launch_args(use_continue=True) == "--continue"

    def test_resume_wins_over_continue(self, kimi: KimiProvider) -> None:
        assert kimi.make_launch_args(resume_id="session_a", use_continue=True) == (
            "-S session_a"
        )

    @pytest.mark.parametrize(
        "bad", ["a; rm -rf /", "$(whoami)", "a b", "`id`", "../etc"]
    )
    def test_rejects_shell_metacharacters(self, kimi: KimiProvider, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid resume_id"):
            kimi.make_launch_args(resume_id=bad)

    def test_yolo_flag_appended(self) -> None:
        assert resolve_launch_command("kimi", approval_mode="yolo") == "kimi --yolo"

    def test_normal_mode_has_no_flag(self) -> None:
        assert resolve_launch_command("kimi") == "kimi"


# ── Detection ────────────────────────────────────────────────────────────


class TestDetection:
    def test_from_pane_command(self) -> None:
        # kimi is a native binary, so the pane reports it directly.
        assert detect_provider_from_command("kimi") == "kimi"

    def test_from_absolute_path(self) -> None:
        assert detect_provider_from_command("/home/u/.kimi-code/bin/kimi") == "kimi"

    def test_kimi_code_argv0_variant(self) -> None:
        """A resumed kimi re-execs with argv[0] == "kimi-code"."""
        assert detect_provider_from_command("kimi-code") == "kimi"
        assert classify_provider_from_argv(["kimi-code", "-S", "session_a"]) == "kimi"

    @pytest.mark.parametrize("cmd", ["kimono", "kimi_other", "akimi"])
    def test_does_not_match_unrelated_command(self, cmd: str) -> None:
        assert detect_provider_from_command(cmd) == ""

    def test_from_foreground_argv(self) -> None:
        argv = ["/home/u/.kimi-code/bin/kimi", "--yolo"]
        assert classify_provider_from_argv(argv) == "kimi"

    def test_argv_skips_wrapper_tokens(self) -> None:
        assert classify_provider_from_argv(["env", "kimi", "--yolo"]) == "kimi"

    def test_from_transcript_path(self) -> None:
        path = "/home/u/.kimi-code/sessions/wd_x_1/session_abc/agents/main/wire.jsonl"
        assert detect_provider_from_transcript_path(path) == "kimi"

    def test_transcript_path_does_not_shadow_claude(self) -> None:
        path = "/home/u/.claude/projects/foo/bar.jsonl"
        assert detect_provider_from_transcript_path(path) == "claude"


# ── Transcript format helpers ────────────────────────────────────────────


class TestEpochConversion:
    def test_converts_milliseconds(self) -> None:
        assert epoch_ms_to_iso(1785613222039) == "2026-08-01T19:40:22.039000+00:00"

    @pytest.mark.parametrize("bad", [None, "1785613222039", True, 0, 1785613222])
    def test_rejects_non_millisecond_values(self, bad: object) -> None:
        # A seconds-based value would silently render as 1970.
        assert epoch_ms_to_iso(bad) is None


class TestExtractText:
    def test_block_list(self) -> None:
        blocks = [{"type": "text", "text": "olá "}, {"type": "text", "text": "mundo"}]
        assert extract_text(blocks) == "olá mundo"

    def test_plain_string(self) -> None:
        assert extract_text("plain") == "plain"

    def test_skips_non_text_blocks(self) -> None:
        blocks = [{"type": "image", "url": "x"}, {"type": "text", "text": "keep"}]
        assert extract_text(blocks) == "keep"

    @pytest.mark.parametrize("bad", [None, 42, {"type": "text"}])
    def test_non_list_non_string(self, bad: object) -> None:
        assert extract_text(bad) == ""


class TestToolCallSummary:
    @pytest.mark.parametrize(
        ("name", "args", "expected_fragment"),
        [
            ("Bash", {"command": "ls -la"}, "ls -la"),
            ("Read", {"path": "src/app.py"}, "src/app.py"),
            ("Write", {"path": "a.txt", "content": "x"}, "a.txt"),
            ("Grep", {"pattern": "TODO"}, "TODO"),
            ("WebSearch", {"query": "kimi cli"}, "kimi cli"),
            ("FetchURL", {"url": "https://x.dev"}, "https://x.dev"),
        ],
    )
    def test_prefers_tool_argument(
        self, name: str, args: dict, expected_fragment: str
    ) -> None:
        assert expected_fragment in tool_call_summary(name, args)

    def test_falls_back_to_description(self) -> None:
        summary = tool_call_summary("TodoList", {}, "Updating the todo list")
        assert "Updating the todo list" in summary

    def test_falls_back_to_any_string_arg(self) -> None:
        assert "abc" in tool_call_summary("Mystery", {"thing": "abc"})

    def test_bare_name_when_nothing_to_show(self) -> None:
        assert tool_call_summary("Mystery", {}) == "🔧 **mystery**"


class TestToolResultText:
    def test_empty_output(self) -> None:
        assert format_tool_result_text("") == "Done"

    def test_short_output_inline(self) -> None:
        assert format_tool_result_text("a\nb") == "a\nb"

    def test_long_output_becomes_expandable_quote(self) -> None:
        text = format_tool_result_text("\n".join(str(i) for i in range(10)))
        assert "10 lines" in text
        assert "EXPQUOTE_START" in text


class TestNormalizePending:
    def test_keeps_string_values(self) -> None:
        assert normalize_pending({"id1": "Bash"}) == {"id1": "Bash"}

    def test_drops_malformed_entries(self) -> None:
        assert normalize_pending({"id1": ("Bash", "Bash"), 2: "x"}) == {}

    @pytest.mark.parametrize("bad", [None, "x", 5])
    def test_non_dict(self, bad: object) -> None:
        assert normalize_pending(bad) == {}


# ── Record parsers ───────────────────────────────────────────────────────


class TestParseTurnPrompt:
    def test_user_turn(self) -> None:
        record = {
            "type": "turn.prompt",
            "input": [{"type": "text", "text": "olá"}],
            "origin": {"kind": "user"},
            "time": 1785613222039,
        }
        msg = parse_turn_prompt(record)
        assert msg is not None
        assert msg.role == "user"
        assert msg.content_type == "text"
        assert msg.text == "olá"
        assert msg.timestamp == "2026-08-01T19:40:22.039000+00:00"

    def test_non_user_origin_skipped(self) -> None:
        record = {
            "type": "turn.prompt",
            "input": [{"type": "text", "text": "replayed"}],
            "origin": {"kind": "injection"},
        }
        assert parse_turn_prompt(record) is None

    def test_empty_input_skipped(self) -> None:
        assert parse_turn_prompt({"type": "turn.prompt", "input": []}) is None


class TestParseContentPart:
    def test_assistant_text(self) -> None:
        msg = parse_content_part({"part": {"type": "text", "text": "pronto"}})
        assert msg is not None
        assert msg.role == "assistant"
        assert msg.content_type == "text"
        assert msg.text == "pronto"

    def test_thinking(self) -> None:
        msg = parse_content_part({"part": {"type": "think", "think": "hmm"}})
        assert msg is not None
        assert msg.content_type == "thinking"
        assert msg.text == "hmm"

    def test_blank_part_skipped(self) -> None:
        assert parse_content_part({"part": {"type": "think", "think": "  "}}) is None

    def test_unknown_part_type(self) -> None:
        assert parse_content_part({"part": {"type": "image", "url": "x"}}) is None

    def test_missing_part(self) -> None:
        assert parse_content_part({}) is None


class TestToolCallResultPairing:
    def test_call_then_result_share_id(self) -> None:
        pending: dict[str, str] = {}
        call = parse_tool_call(
            {
                "type": "tool.call",
                "uuid": "tool_1",
                "toolCallId": "tool_1",
                "name": "Bash",
                "args": {"command": "ls"},
                "description": "Running: ls",
            },
            pending,
        )
        assert call is not None
        assert call.content_type == "tool_use"
        assert call.tool_use_id == "tool_1"
        assert call.tool_name == "Bash"
        assert pending == {"tool_1": "Bash"}

        result = parse_tool_result(
            {
                "type": "tool.result",
                "parentUuid": "tool_1",
                "toolCallId": "tool_1",
                "result": {"output": "a.txt"},
            },
            pending,
        )
        assert result is not None
        assert result.content_type == "tool_result"
        assert result.tool_use_id == "tool_1"
        assert result.tool_name == "Bash"
        assert result.text == "a.txt"
        assert pending == {}

    def test_result_without_matching_call(self) -> None:
        result = parse_tool_result(
            {"toolCallId": "orphan", "result": {"output": "x"}}, {}
        )
        assert result is not None
        assert result.tool_name == "unknown"

    def test_result_falls_back_to_native_tool_name(self) -> None:
        result = parse_tool_result(
            {"toolCallId": "x", "toolName": "Read", "result": {"output": "y"}}, {}
        )
        assert result is not None
        assert result.tool_name == "Read"

    def test_error_result(self) -> None:
        result = parse_tool_result(
            {"toolCallId": "x", "result": {"error": "boom"}}, {"x": "Bash"}
        )
        assert result is not None
        assert result.text == "Error: boom"

    def test_system_note_is_not_relayed(self) -> None:
        result = parse_tool_result(
            {
                "toolCallId": "x",
                "result": {"output": "1\tola", "note": "<system>1 line read</system>"},
            },
            {"x": "Read"},
        )
        assert result is not None
        assert "<system>" not in result.text

    def test_call_falls_back_to_uuid_when_no_call_id(self) -> None:
        pending: dict[str, str] = {}
        call = parse_tool_call({"uuid": "u1", "name": "Read", "args": {}}, pending)
        assert call is not None
        assert call.tool_use_id == "u1"
        assert pending == {"u1": "Read"}


# ── Batch parsing ────────────────────────────────────────────────────────


def _loop(event: dict, time_ms: int = 1785613222039) -> dict:
    return {"type": "context.append_loop_event", "event": event, "time": time_ms}


class TestParseTranscriptEntries:
    def test_full_turn(self, kimi: KimiProvider) -> None:
        entries = [
            {"type": "metadata", "protocol_version": "1.4"},
            {
                "type": "turn.prompt",
                "input": [{"type": "text", "text": "liste"}],
                "origin": {"kind": "user"},
                "time": 1785613222039,
            },
            _loop({"type": "step.begin", "step": 1}),
            _loop({"type": "content.part", "part": {"type": "think", "think": "ok"}}),
            _loop(
                {
                    "type": "tool.call",
                    "toolCallId": "t1",
                    "name": "Bash",
                    "args": {"command": "ls"},
                }
            ),
            _loop(
                {
                    "type": "tool.result",
                    "toolCallId": "t1",
                    "result": {"output": "a.txt"},
                }
            ),
            _loop({"type": "content.part", "part": {"type": "text", "text": "feito"}}),
            _loop({"type": "step.end", "finishReason": "stop"}),
        ]
        messages, pending = kimi.parse_transcript_entries(entries, {})
        assert [(m.role, m.content_type) for m in messages] == [
            ("user", "text"),
            ("assistant", "thinking"),
            ("assistant", "tool_use"),
            ("assistant", "tool_result"),
            ("assistant", "text"),
        ]
        assert pending == {}

    def test_append_message_is_ignored(self, kimi: KimiProvider) -> None:
        """context.append_message duplicates the prompt and carries injections."""
        entries = [
            {
                "type": "turn.prompt",
                "input": [{"type": "text", "text": "olá"}],
                "origin": {"kind": "user"},
            },
            {
                "type": "context.append_message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "olá"}],
                    "origin": {"kind": "user"},
                },
            },
            {
                "type": "context.append_message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "<system-reminder>x"}],
                    "origin": {"kind": "injection", "variant": "todo_list_reminder"},
                },
            },
        ]
        messages, _ = kimi.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "olá"

    def test_pending_carries_across_batches(self, kimi: KimiProvider) -> None:
        first = [
            _loop(
                {
                    "type": "tool.call",
                    "toolCallId": "t9",
                    "name": "Grep",
                    "args": {"pattern": "x"},
                }
            )
        ]
        _, pending = kimi.parse_transcript_entries(first, {})
        assert pending == {"t9": "Grep"}

        second = [
            _loop({"type": "tool.result", "toolCallId": "t9", "result": {"output": ""}})
        ]
        messages, pending = kimi.parse_transcript_entries(second, pending)
        assert messages[0].tool_name == "Grep"
        assert pending == {}

    def test_bookkeeping_records_produce_nothing(self, kimi: KimiProvider) -> None:
        entries = [
            {"type": "llm.request", "model": "k3"},
            {"type": "usage.record", "model": "k3"},
            {"type": "permission.record_approval_result", "toolName": "Bash"},
            {"type": "tools.set_active_tools", "names": ["Bash"]},
            _loop({"type": "step.begin"}),
        ]
        messages, _ = kimi.parse_transcript_entries(entries, {})
        assert messages == []


class TestHistoryEntries:
    def test_user_turn_flagged(self, kimi: KimiProvider) -> None:
        entry = {"type": "turn.prompt", "input": [{"type": "text", "text": "hi"}]}
        assert kimi.is_user_transcript_entry(entry) is True

    def test_injected_turn_not_flagged(self, kimi: KimiProvider) -> None:
        entry = {"type": "turn.prompt", "origin": {"kind": "injection"}}
        assert kimi.is_user_transcript_entry(entry) is False

    def test_loop_event_not_flagged(self, kimi: KimiProvider) -> None:
        assert kimi.is_user_transcript_entry(_loop({"type": "step.begin"})) is False

    def test_history_renders_user_and_assistant_text(self, kimi: KimiProvider) -> None:
        prompt = {
            "type": "turn.prompt",
            "input": [{"type": "text", "text": "pergunta"}],
            "origin": {"kind": "user"},
        }
        reply = _loop({"type": "content.part", "part": {"type": "text", "text": "r"}})
        assert kimi.parse_history_entry(prompt).text == "pergunta"
        assert kimi.parse_history_entry(reply).text == "r"

    def test_history_skips_thinking_and_tools(self, kimi: KimiProvider) -> None:
        think = _loop({"type": "content.part", "part": {"type": "think", "think": "t"}})
        call = _loop({"type": "tool.call", "toolCallId": "1", "name": "Bash"})
        assert kimi.parse_history_entry(think) is None
        assert kimi.parse_history_entry(call) is None


# ── Discovery ────────────────────────────────────────────────────────────


def _write_session(
    home: Path, workspace: str, session_id: str, work_dir: Path, body: str = "{}\n"
) -> Path:
    session_dir = home / ".kimi-code" / "sessions" / workspace / session_id
    transcript = session_dir / "agents" / "main" / "wire.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(body, encoding="utf-8")

    index = home / ".kimi-code" / "session_index.jsonl"
    with index.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "sessionId": session_id,
                    "sessionDir": str(session_dir),
                    "workDir": str(work_dir),
                }
            )
            + "\n"
        )
    return transcript


class TestPaths:
    def test_index_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert session_index_path() == tmp_path / ".kimi-code" / "session_index.jsonl"

    def test_transcript_path(self) -> None:
        assert transcript_for_session_dir("/s/x") == Path("/s/x/agents/main/wire.jsonl")


class TestDiscoverTranscript:
    def test_finds_matching_workdir(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        transcript = _write_session(tmp_path, "wd_proj_1", "session_a", project)

        event = kimi.discover_transcript(str(project), "ccgram:@1")
        assert event is not None
        assert event.session_id == "session_a"
        assert event.transcript_path == str(transcript)
        assert event.window_key == "ccgram:@1"

    def test_ignores_other_workdirs(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        other = tmp_path / "other"
        other.mkdir()
        wanted = tmp_path / "wanted"
        wanted.mkdir()
        _write_session(tmp_path, "wd_other_1", "session_other", other)

        assert kimi.discover_transcript(str(wanted), "ccgram:@1") is None

    def test_picks_freshest_transcript(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        old = _write_session(tmp_path, "wd_proj_1", "session_old", project)
        new = _write_session(tmp_path, "wd_proj_1", "session_new", project)
        import os

        os.utime(old, (1, 1))

        event = kimi.discover_transcript(str(project), "ccgram:@1")
        assert event is not None
        assert event.session_id == "session_new"
        assert event.transcript_path == str(new)

    def test_stale_transcript_rejected(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        transcript = _write_session(tmp_path, "wd_proj_1", "session_a", project)
        import os

        os.utime(transcript, (1, 1))

        assert kimi.discover_transcript(str(project), "ccgram:@1") is None

    def test_max_age_zero_disables_staleness(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        transcript = _write_session(tmp_path, "wd_proj_1", "session_a", project)
        import os

        os.utime(transcript, (1, 1))

        event = kimi.discover_transcript(str(project), "ccgram:@1", max_age=0)
        assert event is not None

    def test_missing_index_returns_none(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert kimi.discover_transcript(str(tmp_path), "ccgram:@1") is None

    def test_corrupt_index_lines_skipped(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        _write_session(tmp_path, "wd_proj_1", "session_a", project)
        index = tmp_path / ".kimi-code" / "session_index.jsonl"
        index.write_text(
            "not json\n{}\n" + index.read_text(encoding="utf-8"), encoding="utf-8"
        )

        event = kimi.discover_transcript(str(project), "ccgram:@1")
        assert event is not None
        assert event.session_id == "session_a"

    def test_index_entry_without_transcript_skipped(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        project = tmp_path / "proj"
        project.mkdir()
        index = tmp_path / ".kimi-code"
        index.mkdir(parents=True, exist_ok=True)
        (index / "session_index.jsonl").write_text(
            json.dumps(
                {
                    "sessionId": "ghost",
                    "sessionDir": str(tmp_path / "gone"),
                    "workDir": str(project),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert kimi.discover_transcript(str(project), "ccgram:@1") is None

    def test_empty_cwd(self, kimi: KimiProvider) -> None:
        assert kimi.discover_transcript("", "ccgram:@1") is None


# ── Terminal chrome ──────────────────────────────────────────────────────


IDLE_PANE = (
    " ● Feito.\n"
    " ╭──────────────────────╮\n"
    " │ >                    │\n"
    " ╰──────────────────────╯\n"
    " K3 thinking: high  …/proj                /init: generate AGENTS.md\n"
    "                                          context: 9% (21k/256k)\n"
)

BUSY_MOON_PANE = (
    " ✨ Liste os arquivos\n"
    "  🌘 · Tip: ctrl-s to add guidance without waiting for the turn to finish\n"
    " ╭──────────────────────╮\n"
    " │ >                    │\n"
    " ╰──────────────────────╯\n"
    " K3 thinking: high  …/proj\n"
    "                                          context: 9% (20.5k/256k)\n"
)

BUSY_BRAILLE_PANE = BUSY_MOON_PANE.replace(
    "  🌘 · Tip: ctrl-s to add guidance without waiting for the turn to finish",
    "  ⠹ working... · Tip: ! to run a shell command",
)

BASH_PERMISSION_PANE = (
    " ● Running a command\n"
    "   $ ls -la\n"
    f" {RULE}\n"
    "   ▶ Run this command?\n"
    "\n"
    "   cwd: /proj\n"
    "   $ ls -la\n"
    "\n"
    "   ▶ 1. Approve once\n"
    "     2. Approve for this session\n"
    "     3. Reject\n"
    "     4. Reject with feedback\n"
    "\n"
    "   ↑/↓ select · 1/2/3/4 choose · ↵ confirm\n"
    f" {RULE}\n"
    " K3 thinking: high  …/proj\n"
    "                                          context: 9% (20.5k/256k)\n"
)

WRITE_PERMISSION_PANE = BASH_PERMISSION_PANE.replace(
    "▶ Run this command?", "▶ Write this file?"
).replace("↵ confirm", "↵ confirm · ctrl+e preview")

PERMISSION_PICKER_PANE = (
    f" {RULE}\n"
    "  Select permission mode\n"
    "  ↑↓ navigate · Enter select · Esc cancel\n"
    "   ❯ Manual ← current\n"
    "     Approve every action yourself.\n"
    "     YOLO\n"
    "     Auto-approve tool actions, but the agent may still ask questions.\n"
    f" {RULE}\n"
    " yolo plan  K3 thinking: high  …/proj\n"
    "                                          context: 0% (0/256k)\n"
)

PROSE_FENCE_PANE = (
    f" {RULE}\n"
    "   Plan mode: ON\n"
    "   Plan will be created here: /home/u/plans/x.md\n"
    f" {RULE}\n"
    " K3 thinking: high  …/proj\n"
    "                                          context: 0% (0/256k)\n"
)


class TestParseTerminalStatus:
    def test_idle_yields_no_status(self, kimi: KimiProvider) -> None:
        assert kimi.parse_terminal_status(IDLE_PANE) is None

    def test_empty_pane(self, kimi: KimiProvider) -> None:
        assert kimi.parse_terminal_status("") is None

    def test_moon_spinner_is_busy(self, kimi: KimiProvider) -> None:
        status = kimi.parse_terminal_status(BUSY_MOON_PANE)
        assert status is not None
        assert status.is_interactive is False
        assert status.display_label == "…working"

    def test_braille_spinner_carries_label(self, kimi: KimiProvider) -> None:
        status = kimi.parse_terminal_status(BUSY_BRAILLE_PANE)
        assert status is not None
        assert status.is_interactive is False
        assert status.display_label == "…working..."

    def test_bash_approval_prompt(self, kimi: KimiProvider) -> None:
        status = kimi.parse_terminal_status(BASH_PERMISSION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"
        assert status.display_label == "Run this command?"
        assert "1. Approve once" in status.raw_text

    def test_write_approval_prompt(self, kimi: KimiProvider) -> None:
        status = kimi.parse_terminal_status(WRITE_PERMISSION_PANE)
        assert status is not None
        assert status.ui_type == "PermissionPrompt"
        assert status.display_label == "Write this file?"

    def test_menu_picker_is_selection_not_permission(self, kimi: KimiProvider) -> None:
        """Option text mentioning "Approve" must not make a picker an approval."""
        status = kimi.parse_terminal_status(PERMISSION_PICKER_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"
        assert status.display_label == "Select permission mode"

    def test_prose_fence_is_not_interactive(self, kimi: KimiProvider) -> None:
        assert kimi.parse_terminal_status(PROSE_FENCE_PANE) is None

    def test_spinner_in_scrollback_is_not_busy(self, kimi: KimiProvider) -> None:
        pane = (
            "  🌘 · Tip: stale frame\n"
            + ("\n".join(["   output"] * 12))
            + ("\n K3 thinking: high  …/proj\n")
        )
        assert kimi.parse_terminal_status(pane) is None

    def test_prompt_wins_over_spinner(self, kimi: KimiProvider) -> None:
        pane = BUSY_MOON_PANE + BASH_PERMISSION_PANE
        status = kimi.parse_terminal_status(pane)
        assert status is not None
        assert status.is_interactive is True


class TestParseStatusModes:
    @pytest.mark.parametrize(
        ("bar", "expected"),
        [
            (" K3 thinking: high  …/proj", None),
            (" yolo  K3 thinking: high  …/proj", "YOLO"),
            (" plan  K3 thinking: high  …/proj", "Plan"),
            (" yolo plan  K3 thinking: high  …/proj", "YOLO Plan"),
            (" auto plan  K3 thinking: high  …/proj", "Auto Plan"),
        ],
    )
    def test_modes(self, bar: str, expected: str | None) -> None:
        pane = f"{bar}\n                       context: 0% (0/256k)\n"
        assert parse_status_modes(pane) == expected

    def test_no_status_bar(self) -> None:
        assert parse_status_modes("just some text\n") is None

    def test_reads_from_full_pane(self) -> None:
        assert parse_status_modes(PERMISSION_PICKER_PANE) == "YOLO Plan"


class TestScrapeCurrentMode:
    async def test_returns_mode_label(self, kimi: KimiProvider) -> None:
        async def fake_capture(_window_id: str) -> str:
            return PERMISSION_PICKER_PANE

        assert await kimi.scrape_current_mode("@1", capture_fn=fake_capture) == (
            "YOLO Plan"
        )

    async def test_no_window_id(self, kimi: KimiProvider) -> None:
        assert await kimi.scrape_current_mode("") is None

    async def test_capture_failure_is_swallowed(self, kimi: KimiProvider) -> None:
        async def boom(_window_id: str) -> str:
            raise OSError("no pane")

        assert await kimi.scrape_current_mode("@1", capture_fn=boom) is None

    async def test_empty_capture(self, kimi: KimiProvider) -> None:
        async def empty(_window_id: str) -> str:
            return ""

        assert await kimi.scrape_current_mode("@1", capture_fn=empty) is None


# ── Status snapshot & commands ───────────────────────────────────────────


class TestStatusSnapshot:
    def test_snapshot_shortens_session_id(
        self, kimi: KimiProvider, tmp_path: Path
    ) -> None:
        transcript = tmp_path / "wire.jsonl"
        transcript.write_text("{}\n", encoding="utf-8")
        snapshot = kimi.build_status_snapshot(
            str(transcript),
            display_name="proj",
            session_id="session_24dbec47-d2ee-410f",
            cwd="/proj",
        )
        assert snapshot is not None
        assert "[proj]" in snapshot
        assert "24dbec47" in snapshot
        assert "session_24dbec47" not in snapshot

    def test_missing_transcript(self, kimi: KimiProvider, tmp_path: Path) -> None:
        assert kimi.build_status_snapshot(str(tmp_path / "nope.jsonl")) is None

    def test_has_output_since(self, kimi: KimiProvider, tmp_path: Path) -> None:
        transcript = tmp_path / "wire.jsonl"
        transcript.write_text("x" * 100, encoding="utf-8")
        assert kimi.has_output_since(str(transcript), 50) is True
        assert kimi.has_output_since(str(transcript), 100) is False

    def test_has_output_since_missing_file(self, kimi: KimiProvider) -> None:
        assert kimi.has_output_since("/nope/wire.jsonl", 0) is False


class TestDiscoverCommands:
    def test_builtins_present(self, kimi: KimiProvider, tmp_path: Path) -> None:
        names = {c.name for c in kimi.discover_commands(str(tmp_path))}
        assert "/new" in names
        assert "/model" in names
        assert "/yolo" in names

    def test_workspace_skills_discovered(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        skill = tmp_path / ".kimi" / "skills" / "deploy-app"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: deploy-app\ndescription: Ship it\n---\nbody\n",
            encoding="utf-8",
        )

        commands = kimi.discover_commands(str(tmp_path))
        deploy = next(c for c in commands if c.name == "/deploy-app")
        assert deploy.description == "Ship it"
        assert deploy.source == "skill"

    def test_skill_without_frontmatter_gets_fallback(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        skill = tmp_path / ".kimi" / "skills" / "bare"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("just text\n", encoding="utf-8")

        commands = kimi.discover_commands(str(tmp_path))
        bare = next(c for c in commands if c.name == "/bare")
        assert "bare" in bare.description

    def test_missing_skill_dirs_are_fine(
        self, kimi: KimiProvider, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        assert kimi.discover_commands(str(tmp_path / "nowhere"))
