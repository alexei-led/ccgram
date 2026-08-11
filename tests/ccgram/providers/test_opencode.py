"""Unit tests for the OpenCode provider (SQLite-backed transcripts)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ccgram.providers import (
    detect_provider_from_command,
    detect_provider_from_transcript_path,
    has_yolo_mode,
    resolve_launch_command,
)
from ccgram.providers.base import AgentMessage, StatusUpdate
from ccgram.providers.opencode import (
    OpenCodeProvider,
    resolve_opencode_db_path,
    resolve_opencode_mirror_root,
)
from ccgram.providers.process_detection import classify_provider_from_args


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, title TEXT, directory TEXT, model TEXT,
            cost REAL, time_archived INTEGER, time_updated INTEGER
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT
        );
        CREATE TABLE event (
            id TEXT PRIMARY KEY, aggregate_id TEXT, seq INTEGER,
            type TEXT, data TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_event(
    conn: sqlite3.Connection,
    session_id: str,
    seq: int,
    etype: str,
    data: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO event (id, aggregate_id, seq, type, data) VALUES (?, ?, ?, ?, ?)",
        (f"evt_{seq}", session_id, seq, etype, json.dumps(data, ensure_ascii=False)),
    )


def _message_event(session_id: str, message_id: str, role: str) -> dict[str, Any]:
    return {
        "sessionID": session_id,
        "info": {"id": message_id, "sessionID": session_id, "role": role},
    }


def _part_event(session_id: str, part: dict[str, Any]) -> dict[str, Any]:
    return {"sessionID": session_id, "part": part}


def _seed_session(
    db: Path,
    session_id: str,
    *,
    cwd: str = "/tmp/repo",
    title: str = "Fix tests",
) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO session (id, title, directory, model, cost, time_archived, time_updated) "
        "VALUES (?, ?, ?, ?, ?, NULL, ?)",
        (
            session_id,
            title,
            cwd,
            "anthropic/claude-sonnet-4-5",
            0.0123,
            1_700_000_000_000,
        ),
    )
    conn.commit()
    return conn


def _seed_events(conn: sqlite3.Connection, session_id: str) -> None:
    # user message + text part (no time — user prompt)
    _insert_event(
        conn,
        session_id,
        1,
        "message.updated.1",
        _message_event(session_id, "msg_user", "user"),
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
        ("msg_user", session_id, json.dumps({"role": "user"})),
    )
    _insert_event(
        conn,
        session_id,
        2,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_user",
                "sessionID": session_id,
                "messageID": "msg_user",
                "type": "text",
                "text": "explain this",
            },
        ),
    )
    # assistant message + settled text part (with time.end)
    _insert_event(
        conn,
        session_id,
        3,
        "message.updated.1",
        _message_event(session_id, "msg_asst", "assistant"),
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
        ("msg_asst", session_id, json.dumps({"role": "assistant"})),
    )
    _insert_event(
        conn,
        session_id,
        4,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_asst",
                "sessionID": session_id,
                "messageID": "msg_asst",
                "type": "text",
                "text": "sure thing",
                "time": {"start": 1, "end": 2},
            },
        ),
    )
    # assistant reasoning part (should be skipped)
    _insert_event(
        conn,
        session_id,
        5,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_reason",
                "sessionID": session_id,
                "messageID": "msg_asst",
                "type": "reasoning",
                "text": "hmm",
            },
        ),
    )
    # tool part: running then completed
    _insert_event(
        conn,
        session_id,
        6,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_tool",
                "sessionID": session_id,
                "messageID": "msg_asst",
                "type": "tool",
                "callID": "call_1",
                "tool": "bash",
                "state": {"status": "running", "input": {"command": "ls"}},
            },
        ),
    )
    _insert_event(
        conn,
        session_id,
        7,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_tool",
                "sessionID": session_id,
                "messageID": "msg_asst",
                "type": "tool",
                "callID": "call_1",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "output": "total 42",
                },
            },
        ),
    )
    # step boundaries (should be skipped)
    _insert_event(
        conn,
        session_id,
        8,
        "message.part.updated.1",
        _part_event(
            session_id,
            {
                "id": "prt_step",
                "sessionID": session_id,
                "messageID": "msg_asst",
                "type": "step-start",
            },
        ),
    )
    conn.commit()


@pytest.fixture()
def provider_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[OpenCodeProvider, Path, Path]:
    db = tmp_path / "opencode.db"
    _make_db(db)
    mirror_root = tmp_path / "mirrors"
    monkeypatch.setenv("CCGRAM_OPENCODE_DB", str(db))
    monkeypatch.setenv("CCGRAM_OPENCODE_DATA_DIR", str(mirror_root))
    return OpenCodeProvider(), db, mirror_root


def _provider(db: Path, mirror_root: Path) -> OpenCodeProvider:
    return OpenCodeProvider()


class TestCapabilities:
    def test_caps(self) -> None:
        caps = OpenCodeProvider().capabilities
        assert caps.name == "opencode"
        assert caps.launch_command == "opencode"
        assert caps.supports_hook is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_status_snapshot is True
        assert caps.supports_incremental_read is False
        assert caps.uses_pyte_status_parsing is False
        assert caps.has_yolo_confirmation is False
        assert "/" not in "".join(caps.builtin_commands)

    def test_yolo_flag_registered(self) -> None:
        assert has_yolo_mode("opencode") is True
        assert (
            resolve_launch_command("opencode", approval_mode="yolo")
            == "opencode --auto"
        )
        assert resolve_launch_command("opencode") == "opencode"

    def test_detection(self) -> None:
        assert detect_provider_from_command("opencode") == "opencode"
        assert classify_provider_from_args("opencode") == "opencode"
        assert (
            detect_provider_from_transcript_path(
                "/home/user/.ccgram/opencode/ses_abc123.jsonl"
            )
            == "opencode"
        )


class TestLaunchArgs:
    def test_fresh(self) -> None:
        assert OpenCodeProvider().make_launch_args() == ""

    def test_resume(self) -> None:
        assert (
            OpenCodeProvider().make_launch_args(resume_id="ses_abc123")
            == "--session ses_abc123"
        )

    def test_resume_invalid_rejected(self) -> None:
        with pytest.raises(ValueError):
            OpenCodeProvider().make_launch_args(resume_id="rm -rf /")

    def test_continue(self) -> None:
        assert OpenCodeProvider().make_launch_args(use_continue=True) == "--continue"


class TestDiscover:
    def test_no_match_returns_none(self, provider_env: tuple) -> None:
        provider, _db, _mirror = provider_env
        assert provider.discover_transcript("/nonexistent", "ccgram:@0") is None

    def test_discover_creates_mirror_with_meta(self, provider_env: tuple) -> None:
        provider, db, mirror_root = provider_env
        conn = _seed_session(db, "ses_1", cwd="/tmp/repo")
        conn.close()
        event = provider.discover_transcript("/tmp/repo", "ccgram:@1")
        assert event is not None
        assert event.session_id == "ses_1"
        assert event.cwd == "/tmp/repo"
        mirror = mirror_root / "ses_1.jsonl"
        assert mirror.exists()
        first = json.loads(mirror.read_text(encoding="utf-8").splitlines()[0])
        assert first["type"] == "session_meta"
        assert first["session_id"] == "ses_1"

    def test_resumable_sessions_filter_cwd(self, provider_env: tuple) -> None:
        provider, db, _mirror = provider_env
        conn = sqlite3.connect(db)
        for i, cwd in enumerate(("/tmp/repo", "/tmp/other", "/tmp/repo")):
            conn.execute(
                "INSERT INTO session (id, title, directory, model, cost, time_archived, time_updated) "
                "VALUES (?, ?, ?, NULL, 0, NULL, ?)",
                (f"ses_{i}", f"title {i}", cwd, 1_700_000_000_000 + i),
            )
        conn.execute(
            "INSERT INTO session (id, title, directory, model, cost, time_archived, time_updated) "
            "VALUES (?, ?, ?, NULL, 0, 1, 1)",
            ("ses_archived", "archived", "/tmp/repo"),
        )
        conn.commit()
        conn.close()
        all_sessions = provider.discover_resumable_sessions()
        assert [s.session_id for s in all_sessions] == ["ses_2", "ses_1", "ses_0"]
        assert "ses_archived" not in [s.session_id for s in all_sessions]
        repo_only = provider.discover_resumable_sessions(cwd="/tmp/repo")
        assert [s.session_id for s in repo_only] == ["ses_2", "ses_0"]
        limited = provider.discover_resumable_sessions(limit=1)
        assert len(limited) == 1 and limited[0].session_id == "ses_2"


class TestTranscriptReading:
    def test_sync_and_parse(self, provider_env: tuple) -> None:
        provider, db, mirror_root = provider_env
        conn = _seed_session(db, "ses_1")
        _seed_events(conn, "ses_1")
        conn.close()
        mirror = mirror_root / "ses_1.jsonl"

        # initial full sync consumes history without emitting messages
        entries, offset = provider.read_transcript_file(str(mirror), 0)
        assert offset == 8
        assert {e["type"] for e in entries} == {"opencode_part"}

        # parse the synced entries
        messages, pending = provider.parse_transcript_entries(entries, {})
        texts = [(m.role, m.content_type, m.text) for m in messages]
        assert ("user", "text", "explain this") in texts
        assert ("assistant", "text", "sure thing") in texts
        # reasoning and step parts are skipped
        assert not any(t == "hmm" for _r, _t, t in texts)
        # tool use emitted on running, result on completed
        tool_msgs = [(m.content_type, m.tool_name, m.tool_use_id) for m in messages]
        assert ("tool_use", "bash", "call_1") in tool_msgs
        assert ("tool_result", "bash", "call_1") in tool_msgs
        assert "call_1" not in pending

    def test_incremental_and_dedupe(self, provider_env: tuple) -> None:
        provider, db, mirror_root = provider_env
        conn = _seed_session(db, "ses_1")
        _seed_events(conn, "ses_1")
        conn.commit()
        mirror = mirror_root / "ses_1.jsonl"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(
            json.dumps(
                {"type": "session_meta", "session_id": "ses_1", "cwd": "/tmp/repo"}
            )
            + "\n",
            encoding="utf-8",
        )

        entries, offset = provider.read_transcript_file(str(mirror), 0)
        assert offset == 8
        first_messages, _ = provider.parse_transcript_entries(entries, {})
        assert len(first_messages) >= 4

        # same events re-read: no duplicates
        entries2, offset2 = provider.read_transcript_file(str(mirror), offset)
        assert entries2 == [] and offset2 == offset
        messages2, _ = provider.parse_transcript_entries(entries2, {})
        assert messages2 == []

        # regression: even an empty read must advance mirror mtime, otherwise
        # TranscriptReader's whole-file gate (current_mtime > last_mtime)
        # closes forever and later events are never polled
        mtime_before = mirror.stat().st_mtime
        entries_noop, _ = provider.read_transcript_file(str(mirror), offset2)
        assert entries_noop == []
        assert mirror.stat().st_mtime > mtime_before

        # new event after cursor
        _insert_event(
            conn,
            "ses_1",
            9,
            "message.part.updated.1",
            _part_event(
                "ses_1",
                {
                    "id": "prt_new",
                    "sessionID": "ses_1",
                    "messageID": "msg_asst",
                    "type": "text",
                    "text": "more",
                    "time": {"start": 1, "end": 2},
                },
            ),
        )
        conn.commit()
        entries3, offset3 = provider.read_transcript_file(str(mirror), offset2)
        assert offset3 == 9
        messages3, _ = provider.parse_transcript_entries(entries3, {})
        assert len(messages3) == 1
        assert messages3[0].text == "more"

    def test_missing_db_fails_closed(self, provider_env: tuple) -> None:
        provider, db, mirror_root = provider_env
        mirror = mirror_root / "ses_x.jsonl"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text("", encoding="utf-8")
        # db deleted → no spin, no crash
        db.unlink()
        entries, offset = provider.read_transcript_file(str(mirror), 7)
        assert entries == [] and offset == 7


class TestEntryHelpers:
    def test_is_user_entry(self) -> None:
        provider = OpenCodeProvider()
        assert (
            provider.is_user_transcript_entry(
                {
                    "type": "opencode_part",
                    "role": "user",
                    "part": {"type": "text", "text": "hi"},
                }
            )
            is True
        )
        assert (
            provider.is_user_transcript_entry(
                {
                    "type": "opencode_part",
                    "role": "assistant",
                    "part": {"type": "text", "text": "hi"},
                }
            )
            is False
        )
        assert provider.is_user_transcript_entry({}) is False

    def test_history_entry(self) -> None:
        provider = OpenCodeProvider()
        msg = provider.parse_history_entry(
            {
                "type": "opencode_part",
                "role": "user",
                "part": {"type": "text", "text": "question"},
            }
        )
        assert msg is not None
        assert isinstance(msg, AgentMessage)
        assert msg.role == "user" and msg.text == "question"
        assert provider.parse_history_entry({"type": "summary"}) is None
        assert (
            provider.parse_history_entry(
                {
                    "type": "opencode_part",
                    "role": "assistant",
                    "part": {"type": "reasoning", "text": "think"},
                }
            )
            is None
        )


class TestStatus:
    def test_permission_prompt_detected(self) -> None:
        provider = OpenCodeProvider()
        status = provider.parse_terminal_status(
            "some output\n! permission requested: bash (ls)\nplease approve or deny"
        )
        assert status is not None
        assert isinstance(status, StatusUpdate)
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_plain_text_not_interactive(self) -> None:
        provider = OpenCodeProvider()
        assert provider.parse_terminal_status("ordinary output\n> prompt") is None

    def test_snapshot_from_db(self, provider_env: tuple) -> None:
        provider, db, _mirror = provider_env
        conn = _seed_session(db, "ses_1")
        conn.close()
        snapshot = provider.build_status_snapshot(
            "/tmp/nonexistent.jsonl",
            display_name="repo",
            session_id="ses_1",
            cwd="/tmp/repo",
        )
        assert snapshot is not None
        assert "repo" in snapshot
        assert "ses_1" in snapshot

    def test_snapshot_missing_db_safe(self, provider_env: tuple) -> None:
        provider, db, _mirror = provider_env
        db.unlink()
        snapshot = provider.build_status_snapshot(
            "/tmp/nonexistent.jsonl", display_name="test"
        )
        assert snapshot is None or "test" in snapshot

    def test_has_output_since_file_semantics(self, tmp_path: Path) -> None:
        provider = OpenCodeProvider()
        transcript = tmp_path / "mirror.jsonl"
        transcript.write_text("line\n", encoding="utf-8")
        assert provider.has_output_since(str(transcript), 0) is True
        size = transcript.stat().st_size
        assert provider.has_output_since(str(transcript), size) is False
        assert provider.has_output_since("/tmp/nonexistent.jsonl", 0) is False


class TestPathResolution:
    def test_db_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db = tmp_path / "custom.db"
        db.write_text("")
        monkeypatch.setenv("CCGRAM_OPENCODE_DB", str(db))
        assert resolve_opencode_db_path() == db

    def test_mirror_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "mirrors"
        monkeypatch.setenv("CCGRAM_OPENCODE_DATA_DIR", str(root))
        assert resolve_opencode_mirror_root() == root
