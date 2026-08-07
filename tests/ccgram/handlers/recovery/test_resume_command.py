import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from ccgram.handlers.callback_data import (
    CB_RESUME_CANCEL,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
)
from ccgram.handlers.recovery.resume_command import (
    ResumeEntry,
    _build_resume_keyboard,
    _relative_time,
    format_session_entry,
    handle_resume_command_callback,
    resume_command,
    scan_all_sessions,
)
from ccgram.handlers.user_state import RESUME_SESSIONS

_RC = "ccgram.handlers.recovery.resume_command"


def _make_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
    thread_id: int = 42,
    text: str = "/resume",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=chat_id)
    msg = MagicMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True
    msg.is_topic_message = True
    update.message = msg
    update.callback_query = None
    return update


def _make_callback_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
    thread_id: int = 42,
    data: str = "",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=chat_id)
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = chat_id
    query.message.message_thread_id = thread_id
    query.message.chat.is_forum = True
    query.message.is_topic_message = True
    update.callback_query = query
    update.message = None
    return update


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    return ctx


class TestScanAllSessions:
    def test_returns_sessions_from_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Fix the bug",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "sess-1"
        assert result[0].summary == "Fix the bug"
        assert result[0].cwd == str(Path("/tmp/myproj").resolve())

    def test_returns_empty_when_projects_path_missing(self, tmp_path) -> None:
        with patch(
            "ccgram.providers.claude._claude_projects_path",
            return_value=tmp_path / "nonexistent",
        ):
            result = scan_all_sessions()

        assert result == []

    def test_deduplicates_by_session_id(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"

        for name in ("proj-a", "proj-b"):
            proj_dir = projects_path / name
            proj_dir.mkdir(parents=True)
            sf = proj_dir / "sess-dup.jsonl"
            sf.write_text('{"type":"summary"}\n')
            index = {
                "originalPath": f"/tmp/{name}",
                "entries": [
                    {
                        "sessionId": "sess-dup",
                        "fullPath": str(sf),
                        "projectPath": f"/tmp/{name}",
                        "summary": f"From {name}",
                    }
                ],
            }
            (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1

    def test_skips_missing_session_files(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-gone",
                    "fullPath": str(proj_dir / "nonexistent.jsonl"),
                    "projectPath": "/tmp/myproj",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert result == []

    def test_uses_session_id_as_summary_fallback(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-abc123.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "a1b2c3d4-0000-0000-0000-abc123000000",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "a1b2c3d4-000"

    def test_sorted_by_mtime_descending(self, tmp_path) -> None:
        import time

        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        old_file = proj_dir / "sess-old.jsonl"
        old_file.write_text('{"type":"summary"}\n')
        time.sleep(0.05)

        new_file = proj_dir / "sess-new.jsonl"
        new_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-old",
                    "fullPath": str(old_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Old session",
                },
                {
                    "sessionId": "sess-new",
                    "fullPath": str(new_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "New session",
                },
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 2
        assert result[0].session_id == "sess-new"
        assert result[1].session_id == "sess-old"

    def test_scans_multiple_projects(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"

        for i, name in enumerate(("proj-a", "proj-b")):
            proj_dir = projects_path / name
            proj_dir.mkdir(parents=True)
            sf = proj_dir / f"sess-{i}.jsonl"
            sf.write_text('{"type":"summary"}\n')
            index = {
                "originalPath": f"/tmp/{name}",
                "entries": [
                    {
                        "sessionId": f"sess-{i}",
                        "fullPath": str(sf),
                        "projectPath": f"/tmp/{name}",
                        "summary": f"Session {i}",
                    }
                ],
            }
            (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 2
        ids = {r.session_id for r in result}
        assert ids == {"sess-0", "sess-1"}

    def test_skips_invalid_json(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "sessions-index.json").write_text("not valid json{{{")

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert result == []

    def test_bare_jsonl_without_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "abc-123.jsonl"
        jsonl.write_text(
            '{"type":"user","cwd":"/tmp/myproj","message":{"content":[{"type":"text","text":"Fix the bug"}]}}\n'
        )

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "abc-123"
        assert result[0].cwd == str(Path("/tmp/myproj").resolve())
        assert result[0].summary == "Fix the bug"

    def test_bare_jsonl_skips_no_cwd(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "no-cwd.jsonl"
        jsonl.write_text('{"type":"file-history-snapshot"}\n')

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert result == []

    def test_bare_jsonl_deduplicates_with_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text(
            '{"type":"user","cwd":"/tmp/myproj","message":{"content":"hi"}}\n'
        )

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "From index",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "From index"

    def test_uses_first_prompt_as_summary_fallback(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-fp.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-fp",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "firstPrompt": "Implement auth",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "Implement auth"

    def test_entries_carry_mtime_from_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        expected_mtime = session_file.stat().st_mtime

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Fix bug",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].mtime == expected_mtime

    def test_bare_jsonl_entries_carry_mtime(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "abc-123.jsonl"
        jsonl.write_text(
            '{"type":"user","cwd":"/tmp/myproj","message":{"content":"hi"}}\n'
        )
        expected_mtime = jsonl.stat().st_mtime

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].mtime == expected_mtime


class TestRelativeTime:
    _NOW = 1_700_000_000.0

    def test_never_when_zero_mtime(self) -> None:
        assert _relative_time(0.0, now=self._NOW) == "never"

    def test_never_when_negative_mtime(self) -> None:
        assert _relative_time(-1.0, now=self._NOW) == "never"

    def test_today_under_24h(self) -> None:
        assert _relative_time(self._NOW - 60, now=self._NOW) == "today"
        assert _relative_time(self._NOW - 3600 * 23, now=self._NOW) == "today"

    def test_yesterday_within_two_days(self) -> None:
        assert _relative_time(self._NOW - 86400 * 1.5, now=self._NOW) == "yesterday"

    def test_n_days_ago(self) -> None:
        assert _relative_time(self._NOW - 86400 * 3, now=self._NOW) == "3d ago"
        assert _relative_time(self._NOW - 86400 * 14, now=self._NOW) == "14d ago"

    def test_future_mtime_treated_as_today(self) -> None:
        assert _relative_time(self._NOW + 60, now=self._NOW) == "today"


class TestFormatSessionEntry:
    _NOW = 1_700_000_000.0

    def test_full_format(self) -> None:
        out = format_session_entry(
            summary="Fix login bug",
            session_id="a1b2c3d4-0000-0000-0000-deadbeefcafe",
            mtime=self._NOW - 60,
            now=self._NOW,
        )
        assert out == "today · Fix login bug · cafe"

    def test_yesterday_label(self) -> None:
        out = format_session_entry(
            summary="Add tests",
            session_id="x1y2z3-0000-1111-2222-3333abcd9999",
            mtime=self._NOW - 86400 * 1.5,
            now=self._NOW,
        )
        assert out.startswith("yesterday · ")
        assert out.endswith(" · 9999")

    def test_n_days_ago_label(self) -> None:
        out = format_session_entry(
            summary="Refactor parser",
            session_id="aaaaaa-bbbb-cccc-dddd-eeeeffff1234",
            mtime=self._NOW - 86400 * 5,
            now=self._NOW,
        )
        assert "5d ago" in out
        assert out.endswith(" · 1234")

    def test_never_label_for_missing_mtime(self) -> None:
        out = format_session_entry(
            summary="Old session",
            session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeffff5678",
            mtime=0.0,
            now=self._NOW,
        )
        assert out.startswith("never · ")
        assert out.endswith(" · 5678")

    def test_summary_truncated_to_40(self) -> None:
        long = "A" * 80
        out = format_session_entry(
            summary=long,
            session_id="abcd",
            mtime=self._NOW,
            now=self._NOW,
        )
        parts = out.split(" · ")
        assert len(parts[1]) == 40

    def test_summary_newlines_collapsed(self) -> None:
        out = format_session_entry(
            summary="line one\nline two\nline three",
            session_id="abcd",
            mtime=self._NOW,
            now=self._NOW,
        )
        assert "\n" not in out
        assert "line one" in out
        assert "line two" not in out

    def test_empty_summary_falls_back_to_session_id_prefix(self) -> None:
        out = format_session_entry(
            summary="",
            session_id="ab12cd34-ef56-7890-aaaa-bbbbccccdddd",
            mtime=self._NOW,
            now=self._NOW,
        )
        parts = out.split(" · ")
        assert parts[1] == "ab12cd34-ef5"

    def test_short_session_id_last4(self) -> None:
        out = format_session_entry(
            summary="x",
            session_id="abc",
            mtime=self._NOW,
            now=self._NOW,
        )
        assert out.endswith(" · abc")

    def test_empty_session_id(self) -> None:
        out = format_session_entry(
            summary="x",
            session_id="",
            mtime=self._NOW,
            now=self._NOW,
        )
        assert out.endswith(" · ????")

    def test_msg_count_appended_when_set(self) -> None:
        out = format_session_entry(
            summary="Fix bug",
            session_id="abcd1234-eeee-ffff-0000-1111deadbeef",
            mtime=self._NOW,
            msg_count=42,
            now=self._NOW,
        )
        assert out.endswith(" · 42 msgs")

    def test_msg_count_omitted_when_none(self) -> None:
        out = format_session_entry(
            summary="Fix bug",
            session_id="abcd",
            mtime=self._NOW,
            msg_count=None,
            now=self._NOW,
        )
        assert "msgs" not in out

    def test_msg_count_omitted_when_zero(self) -> None:
        out = format_session_entry(
            summary="Fix bug",
            session_id="abcd",
            mtime=self._NOW,
            msg_count=0,
            now=self._NOW,
        )
        assert "msgs" not in out


class TestIndexMsgCount:
    def test_pulls_message_count_field(self) -> None:
        from ccgram.handlers.recovery.resume_command import _index_msg_count

        assert _index_msg_count({"messageCount": 7}) == 7

    def test_pulls_msg_count_alias(self) -> None:
        from ccgram.handlers.recovery.resume_command import _index_msg_count

        assert _index_msg_count({"msgCount": 9}) == 9

    def test_returns_none_for_missing_field(self) -> None:
        from ccgram.handlers.recovery.resume_command import _index_msg_count

        assert _index_msg_count({"otherField": 5}) is None

    def test_returns_none_for_zero(self) -> None:
        from ccgram.handlers.recovery.resume_command import _index_msg_count

        assert _index_msg_count({"messageCount": 0}) is None

    def test_returns_none_for_non_int(self) -> None:
        from ccgram.handlers.recovery.resume_command import _index_msg_count

        assert _index_msg_count({"messageCount": "many"}) is None


class TestResumeEntryMsgCount:
    def test_default_msg_count_is_none(self) -> None:
        entry = ResumeEntry("sid", "summary", "/cwd", 1.0)
        assert entry.msg_count is None

    def test_msg_count_round_trips(self) -> None:
        entry = ResumeEntry("sid", "summary", "/cwd", 1.0, msg_count=15)
        assert entry.msg_count == 15

    def test_scan_pulls_msg_count_from_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        sf = proj_dir / "sess-1.jsonl"
        sf.write_text('{"type":"summary"}\n')
        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(sf),
                    "projectPath": "/tmp/myproj",
                    "summary": "Indexed",
                    "messageCount": 23,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(
            "ccgram.providers.claude._claude_projects_path", return_value=projects_path
        ):
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].msg_count == 23


class TestResumeKeyboardMsgCount:
    def test_label_includes_msg_count_when_supplied(self) -> None:
        import time as _time

        recent = _time.time() - 10
        sessions = [
            {
                "session_id": "abcd1234-eeee-ffff-0000-1111deadbeef",
                "summary": "Implement auth",
                "cwd": "/proj/a",
                "mtime": recent,
                "msg_count": 42,
            }
        ]
        kb = _build_resume_keyboard(sessions)
        button = kb.inline_keyboard[1][0]
        assert "42 msgs" in button.text

    def test_label_omits_msg_count_when_missing(self) -> None:
        import time as _time

        recent = _time.time() - 10
        sessions = [
            {
                "session_id": "abcd1234-eeee-ffff-0000-1111deadbeef",
                "summary": "No count",
                "cwd": "/proj/a",
                "mtime": recent,
            }
        ]
        kb = _build_resume_keyboard(sessions)
        button = kb.inline_keyboard[1][0]
        assert "msgs" not in button.text


class TestResumeImprovedToastWordings:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_pick_invalid_index_says_no_longer(
        self,
        _mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}99")
        user_data: dict = {
            RESUME_SESSIONS: [
                {"session_id": "x", "summary": "x", "cwd": "/tmp"},
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        text = query.answer.call_args.kwargs.get(
            "text",
            query.answer.call_args.args[0] if query.answer.call_args.args else "",
        )
        assert "no longer" in text.lower()

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_pick_invalid_value_says_couldnt_read(
        self,
        _mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}notanumber")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        text = query.answer.call_args.kwargs.get(
            "text",
            query.answer.call_args.args[0] if query.answer.call_args.args else "",
        )
        assert "couldn" in text.lower()


class TestBuildResumeKeyboard:
    def _sessions(self, count: int = 3) -> list[dict[str, str]]:
        return [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(count)
        ]

    def test_session_buttons(self) -> None:
        sessions = self._sessions(2)
        kb = _build_resume_keyboard(sessions)
        assert len(kb.inline_keyboard) == 4
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_RESUME_PICK}0"
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_RESUME_PICK}1"

    def test_project_header(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions)
        header = kb.inline_keyboard[0][0]
        assert "proj" in header.text
        assert header.callback_data == "noop"

    def test_cancel_button_present(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions)
        nav_row = kb.inline_keyboard[-1]
        cancel = [b for b in nav_row if b.callback_data == CB_RESUME_CANCEL]
        assert len(cancel) == 1

    def test_no_prev_on_first_page(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions, page=0)
        nav_row = kb.inline_keyboard[-1]
        prev_btns = [
            b
            for b in nav_row
            if isinstance(b.callback_data, str) and CB_RESUME_PAGE in b.callback_data
        ]
        assert len(prev_btns) == 0

    def test_next_button_on_first_page(self) -> None:
        sessions = self._sessions(10)
        kb = _build_resume_keyboard(sessions, page=0)
        nav_row = kb.inline_keyboard[-1]
        next_btns = [
            b
            for b in nav_row
            if isinstance(b.callback_data, str)
            and b.callback_data.startswith(CB_RESUME_PAGE)
        ]
        assert len(next_btns) == 1
        assert "Next" in next_btns[0].text

    def test_prev_button_on_second_page(self) -> None:
        sessions = self._sessions(10)
        kb = _build_resume_keyboard(sessions, page=1)
        nav_row = kb.inline_keyboard[-1]
        prev_btns = [b for b in nav_row if "Prev" in b.text]
        assert len(prev_btns) == 1

    def test_grouped_by_cwd(self) -> None:
        sessions = [
            {"session_id": "s1", "summary": "A", "cwd": "/proj/a"},
            {"session_id": "s2", "summary": "B", "cwd": "/proj/b"},
        ]
        kb = _build_resume_keyboard(sessions)
        headers = [
            row[0] for row in kb.inline_keyboard if row[0].callback_data == "noop"
        ]
        assert len(headers) == 2

    def test_label_uses_format_session_entry(self) -> None:
        import time as _time

        recent = _time.time() - 10
        sessions = [
            {
                "session_id": "abcd1234-eeee-ffff-0000-1111deadbeef",
                "summary": "Implement auth",
                "cwd": "/proj/a",
                "mtime": recent,
            }
        ]
        kb = _build_resume_keyboard(sessions)
        # First row is the project header, second row is the session button.
        button = kb.inline_keyboard[1][0]
        assert "today" in button.text
        assert "Implement auth" in button.text
        assert button.text.endswith(" · beef")

    def test_label_handles_missing_mtime(self) -> None:
        sessions = [
            {
                "session_id": "abcd1234-eeee-ffff-0000-1111deadbe99",
                "summary": "Old",
                "cwd": "/proj/a",
            }
        ]
        kb = _build_resume_keyboard(sessions)
        button = kb.inline_keyboard[1][0]
        assert button.text.startswith("never · ")


class TestResumeCommand:
    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_shows_session_picker(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_scan.return_value = [
            ResumeEntry("sess-1", "Fix bug", "/tmp/proj"),
            ResumeEntry("sess-2", "Add tests", "/tmp/proj"),
        ]

        update = _make_update()
        user_data: dict = {}
        ctx = _make_context(user_data)

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert "Select a session" in mock_safe_reply.call_args.args[1]
        assert RESUME_SESSIONS in user_data
        assert len(user_data[RESUME_SESSIONS]) == 2

    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_no_sessions_shows_message(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_scan.return_value = []

        update = _make_update()
        ctx = _make_context()

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert "No past sessions" in mock_safe_reply.call_args.args[1]

    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=None)
    @patch(f"{_RC}.config")
    async def test_no_topic_rejected(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        update = _make_update()
        ctx = _make_context()

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert "named topic" in mock_safe_reply.call_args.args[1]

    async def test_no_message_returns_early(self) -> None:
        update = MagicMock()
        update.message = None
        ctx = _make_context()

        await resume_command(update, ctx)


class TestResumePickCallback:
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_creates_window_with_resume(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tm.create_window.assert_called_once_with(
            "/tmp/proj",
            agent_args="--resume a1b2c3d4-0000-0000-0000-000000000001",
            launch_command="claude",
        )
        mock_tr.bind_thread.assert_called_once_with(
            100, 42, "@5", window_name="project"
        )

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_unbinds_old_window(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tr.unbind_thread.assert_called_once_with(100, 42)

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_invalid_cwd_fails(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/gone",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = False
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "no longer exists" in mock_safe_edit.call_args.args[1].lower()
        assert RESUME_SESSIONS not in user_data

    async def test_pick_invalid_index_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}99")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()
        assert (
            "no longer"
            in query.answer.call_args.kwargs.get(
                "text",
                query.answer.call_args.args[0] if query.answer.call_args.args else "",
            ).lower()
        )

    async def test_pick_no_sessions_stored_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        ctx = _make_context({})
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    async def test_pick_no_topic_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=None):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_second_session(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}1")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000002",
                    "summary": "Add tests",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tm.create_window.assert_called_once_with(
            "/tmp/proj",
            agent_args="--resume a1b2c3d4-0000-0000-0000-000000000002",
            launch_command="claude",
        )

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_sets_group_chat_id(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -100999)

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_clears_resume_state(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert RESUME_SESSIONS not in user_data

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_create_window_failure(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(False, "Tmux error", None, None)
        )

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "Tmux error" in mock_safe_edit.call_args.args[1]
        assert RESUME_SESSIONS not in user_data

    async def test_pick_invalid_value_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}notanumber")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()


class TestResumePageCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_page_shows_sessions(
        self,
        mock_safe_edit: AsyncMock,
    ) -> None:
        sessions = [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(10)
        ]
        user_data: dict = {RESUME_SESSIONS: sessions}
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}1")
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "Select a session" in mock_safe_edit.call_args.args[1]

    async def test_page_invalid_number_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}abc")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    async def test_page_no_sessions_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}0")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()


class TestResumeCancelCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_cancel_clears_state(
        self,
        mock_safe_edit: AsyncMock,
    ) -> None:
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        update = _make_callback_update(data=CB_RESUME_CANCEL)
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert RESUME_SESSIONS not in user_data
        mock_safe_edit.assert_called_once()
        assert "cancelled" in mock_safe_edit.call_args.args[1].lower()

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_cancel_answers_query(
        self,
        _mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=CB_RESUME_CANCEL)
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once_with("Cancelled")


class TestResumePerWindowProvider:
    @patch(f"{_RC}.get_provider_for_window")
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_uses_per_window_provider_when_bound(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_gpw: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@3"
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999
        mock_gpw.return_value.make_launch_args.return_value = "--resume sess-1"

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "sess-1",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_gpw.assert_called_once_with("@3", provider_name=ANY)

    @patch(f"{_RC}.get_provider")
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_falls_back_to_global_provider_when_unbound(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_gp: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999
        mock_gp.return_value.make_launch_args.return_value = "--resume sess-1"

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "sess-1",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_gp.assert_called_once()
        mock_gp.return_value.make_launch_args.assert_called_once_with(
            resume_id="sess-1"
        )
