import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from ccgram.handlers.callback_data import (
    CB_RESUME_CANCEL,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
)
from ccgram.handlers.recovery.resume_command import (
    ResumeEntry,
    _build_resume_keyboard,
    _index_msg_count,
    _relative_time,
    format_session_entry,
    handle_resume_command_callback,
    resume_command,
    scan_all_sessions,
)
from ccgram.handlers.user_state import RESUME_SESSIONS

_RC = "ccgram.handlers.recovery.resume_command"
_NOW = 1_700_000_000.0
_SID = "a1b2c3d4-0000-0000-0000-000000000001"
_SID2 = "a1b2c3d4-0000-0000-0000-000000000002"


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


def _session(
    session_id: str = _SID,
    summary: str = "Fix bug",
    cwd: str = "/tmp/proj",
    **extra,
) -> dict:
    return {"session_id": session_id, "summary": summary, "cwd": cwd, **extra}


def _toast(query: AsyncMock) -> str:
    """Text of the last ``query.answer`` toast, positional or keyword."""
    call = query.answer.call_args
    assert call is not None
    if call.args:
        return str(call.args[0])
    return str(call.kwargs.get("text", ""))


# ── Claude session discovery on disk ──────────────────────────────────────


@pytest.fixture()
def projects_root(tmp_path: Path):
    """Point Claude session discovery at an (initially absent) tmp root."""
    root = tmp_path / "projects"
    with patch("ccgram.providers.claude._claude_projects_path", return_value=root):
        yield root


def _project(projects_root: Path, name: str = "-tmp-myproj") -> Path:
    proj = projects_root / name
    proj.mkdir(parents=True)
    return proj


def _session_file(proj: Path, name: str, body: str = '{"type":"summary"}\n') -> Path:
    path = proj / f"{name}.jsonl"
    path.write_text(body)
    return path


def _index_entry(session_id: str, path: Path, project_path: str, **extra) -> dict:
    return {
        "sessionId": session_id,
        "fullPath": str(path),
        "projectPath": project_path,
        **extra,
    }


def _write_index(proj: Path, original_path: str, *entries: dict) -> None:
    (proj / "sessions-index.json").write_text(
        json.dumps({"originalPath": original_path, "entries": list(entries)})
    )


def _user_line(cwd: str, text: str = "hi") -> str:
    return (
        json.dumps(
            {
                "type": "user",
                "cwd": cwd,
                "message": {"content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


class TestScanAllSessions:
    def test_returns_sessions_from_index(self, projects_root) -> None:
        proj = _project(projects_root)
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry(
                "sess-1",
                _session_file(proj, "sess-1"),
                "/tmp/myproj",
                summary="Fix the bug",
            ),
        )

        result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "sess-1"
        assert result[0].summary == "Fix the bug"
        assert result[0].cwd == str(Path("/tmp/myproj").resolve())

    def test_returns_empty_when_projects_path_missing(self, projects_root) -> None:
        assert scan_all_sessions() == []

    def test_scans_and_deduplicates_across_projects(self, projects_root) -> None:
        for name in ("proj-a", "proj-b"):
            proj = _project(projects_root, name)
            _write_index(
                proj,
                f"/tmp/{name}",
                _index_entry(
                    "sess-dup",
                    _session_file(proj, "sess-dup"),
                    f"/tmp/{name}",
                    summary=f"From {name}",
                ),
                _index_entry(
                    f"sess-{name}",
                    _session_file(proj, f"sess-{name}"),
                    f"/tmp/{name}",
                ),
            )

        ids = {entry.session_id for entry in scan_all_sessions()}

        assert ids == {"sess-dup", "sess-proj-a", "sess-proj-b"}

    def test_skips_missing_session_files(self, projects_root) -> None:
        proj = _project(projects_root)
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry("sess-gone", proj / "nonexistent.jsonl", "/tmp/myproj"),
        )

        assert scan_all_sessions() == []

    def test_skips_invalid_json(self, projects_root) -> None:
        proj = _project(projects_root)
        (proj / "sessions-index.json").write_text("not valid json{{{")

        assert scan_all_sessions() == []

    @pytest.mark.parametrize(
        ("extra", "expected_summary"),
        [
            ({"summary": "Fix the bug"}, "Fix the bug"),
            ({"firstPrompt": "Implement auth"}, "Implement auth"),
            ({}, "a1b2c3d4-000"),
        ],
        ids=["summary", "first-prompt-fallback", "session-id-fallback"],
    )
    def test_index_summary_precedence(
        self, projects_root, extra: dict, expected_summary: str
    ) -> None:
        proj = _project(projects_root)
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry(
                "a1b2c3d4-0000-0000-0000-abc123000000",
                _session_file(proj, "sess-1"),
                "/tmp/myproj",
                **extra,
            ),
        )

        result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == expected_summary

    def test_sorted_by_mtime_descending(self, projects_root) -> None:
        import time

        proj = _project(projects_root)
        old_file = _session_file(proj, "sess-old")
        time.sleep(0.05)
        new_file = _session_file(proj, "sess-new")
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry("sess-old", old_file, "/tmp/myproj", summary="Old"),
            _index_entry("sess-new", new_file, "/tmp/myproj", summary="New"),
        )

        result = scan_all_sessions()

        assert [entry.session_id for entry in result] == ["sess-new", "sess-old"]

    @pytest.mark.parametrize("indexed", [True, False], ids=["index", "bare-jsonl"])
    def test_entries_carry_file_mtime(self, projects_root, indexed: bool) -> None:
        proj = _project(projects_root)
        session_file = _session_file(proj, "sess-1", _user_line("/tmp/myproj"))
        if indexed:
            _write_index(
                proj,
                "/tmp/myproj",
                _index_entry("sess-1", session_file, "/tmp/myproj", summary="Fix bug"),
            )

        result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].mtime == session_file.stat().st_mtime

    def test_pulls_msg_count_from_index(self, projects_root) -> None:
        proj = _project(projects_root)
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry(
                "sess-1",
                _session_file(proj, "sess-1"),
                "/tmp/myproj",
                summary="Indexed",
                messageCount=23,
            ),
        )

        result = scan_all_sessions()

        assert result[0].msg_count == 23

    def test_bare_jsonl_without_index(self, projects_root) -> None:
        proj = _project(projects_root)
        _session_file(proj, "abc-123", _user_line("/tmp/myproj", "Fix the bug"))

        result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "abc-123"
        assert result[0].cwd == str(Path("/tmp/myproj").resolve())
        assert result[0].summary == "Fix the bug"

    def test_bare_jsonl_skips_file_without_cwd(self, projects_root) -> None:
        proj = _project(projects_root)
        _session_file(proj, "no-cwd", '{"type":"file-history-snapshot"}\n')

        assert scan_all_sessions() == []

    def test_index_summary_wins_over_bare_jsonl(self, projects_root) -> None:
        proj = _project(projects_root)
        session_file = _session_file(proj, "sess-1", _user_line("/tmp/myproj"))
        _write_index(
            proj,
            "/tmp/myproj",
            _index_entry("sess-1", session_file, "/tmp/myproj", summary="From index"),
        )

        result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "From index"


# ── Label rendering ───────────────────────────────────────────────────────


class TestRelativeTime:
    @pytest.mark.parametrize(
        ("mtime", "expected"),
        [
            (0.0, "never"),
            (-1.0, "never"),
            (_NOW - 60, "today"),
            (_NOW + 60, "today"),
            (_NOW - 86400 * 0.99, "today"),
            (_NOW - 86400, "yesterday"),
            (_NOW - 86400 * 1.5, "yesterday"),
            (_NOW - 86400 * 2, "2d ago"),
            (_NOW - 86400 * 14, "14d ago"),
        ],
        ids=[
            "zero",
            "negative",
            "minutes-ago",
            "clock-skew-future",
            "just-under-a-day",
            "exactly-one-day",
            "day-and-a-half",
            "exactly-two-days",
            "two-weeks",
        ],
    )
    def test_relative_time(self, mtime: float, expected: str) -> None:
        assert _relative_time(mtime, now=_NOW) == expected


class TestFormatSessionEntry:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {
                    "summary": "Fix login bug",
                    "session_id": "a1b2c3d4-0000-0000-0000-deadbeefcafe",
                    "mtime": _NOW - 60,
                },
                "today · Fix login bug · cafe",
            ),
            (
                {
                    "summary": "Add tests",
                    "session_id": "x1y2z3-0000-1111-2222-3333abcd9999",
                    "mtime": _NOW - 86400 * 1.5,
                },
                "yesterday · Add tests · 9999",
            ),
            (
                {
                    "summary": "Refactor parser",
                    "session_id": "aaaaaa-bbbb-cccc-dddd-eeeeffff1234",
                    "mtime": _NOW - 86400 * 5,
                },
                "5d ago · Refactor parser · 1234",
            ),
            (
                {
                    "summary": "Old session",
                    "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeffff5678",
                    "mtime": 0.0,
                },
                "never · Old session · 5678",
            ),
            (
                {"summary": "A" * 80, "session_id": "abcd", "mtime": _NOW},
                f"today · {'A' * 40} · abcd",
            ),
            (
                {
                    "summary": "line one\nline two\nline three",
                    "session_id": "abcd",
                    "mtime": _NOW,
                },
                "today · line one · abcd",
            ),
            (
                {
                    "summary": "",
                    "session_id": "ab12cd34-ef56-7890-aaaa-bbbbccccdddd",
                    "mtime": _NOW,
                },
                "today · ab12cd34-ef5 · dddd",
            ),
            (
                {"summary": "", "session_id": "", "mtime": _NOW},
                "today · (unknown) · ????",
            ),
            (
                {"summary": "x", "session_id": "abc", "mtime": _NOW},
                "today · x · abc",
            ),
            (
                {
                    "summary": "Fix bug",
                    "session_id": "abcd1234-eeee-ffff-0000-1111deadbeef",
                    "mtime": _NOW,
                    "msg_count": 42,
                },
                "today · Fix bug · beef · 42 msgs",
            ),
            (
                {
                    "summary": "Fix bug",
                    "session_id": "abcd",
                    "mtime": _NOW,
                    "msg_count": None,
                },
                "today · Fix bug · abcd",
            ),
            (
                {
                    "summary": "Fix bug",
                    "session_id": "abcd",
                    "mtime": _NOW,
                    "msg_count": 0,
                },
                "today · Fix bug · abcd",
            ),
        ],
        ids=[
            "today",
            "yesterday",
            "n-days-ago",
            "never",
            "summary-truncated-to-40",
            "newlines-collapsed",
            "empty-summary-uses-session-id-prefix",
            "empty-session-id",
            "session-id-shorter-than-four",
            "msg-count-appended",
            "msg-count-none",
            "msg-count-zero",
        ],
    )
    def test_format_session_entry(self, kwargs: dict, expected: str) -> None:
        assert format_session_entry(now=_NOW, **kwargs) == expected


class TestIndexMsgCount:
    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ({"messageCount": 7}, 7),
            ({"msgCount": 9}, 9),
            ({"msg_count": 3}, 3),
            ({"messages": 5}, 5),
            ({"otherField": 5}, None),
            ({"messageCount": 0}, None),
            ({"messageCount": -2}, None),
            ({"messageCount": "many"}, None),
        ],
        ids=[
            "messageCount",
            "msgCount-alias",
            "msg_count-alias",
            "messages-alias",
            "missing",
            "zero",
            "negative",
            "non-int",
        ],
    )
    def test_index_msg_count(self, entry: dict, expected: int | None) -> None:
        assert _index_msg_count(entry) == expected


class TestBuildResumeKeyboard:
    def _sessions(self, count: int = 3) -> list[dict]:
        return [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(count)
        ]

    def test_one_button_per_session_under_a_project_header(self) -> None:
        kb = _build_resume_keyboard(self._sessions(2))

        header = kb.inline_keyboard[0][0]
        assert "proj" in header.text
        assert header.callback_data == "noop"
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_RESUME_PICK}0"
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_RESUME_PICK}1"

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

    def test_cancel_button_present(self) -> None:
        kb = _build_resume_keyboard(self._sessions(1))

        nav_row = kb.inline_keyboard[-1]
        assert [b.callback_data for b in nav_row] == [CB_RESUME_CANCEL]

    @pytest.mark.parametrize(
        ("count", "page", "expected_nav"),
        [
            (1, 0, []),
            (10, 0, [f"{CB_RESUME_PAGE}1"]),
            (10, 1, [f"{CB_RESUME_PAGE}0"]),
            (20, 1, [f"{CB_RESUME_PAGE}0", f"{CB_RESUME_PAGE}2"]),
        ],
        ids=["single-page", "first-page", "last-page", "middle-page"],
    )
    def test_pagination_buttons(
        self, count: int, page: int, expected_nav: list[str]
    ) -> None:
        kb = _build_resume_keyboard(self._sessions(count), page=page)

        nav_row = kb.inline_keyboard[-1]
        page_buttons = [
            b.callback_data
            for b in nav_row
            if isinstance(b.callback_data, str)
            and b.callback_data.startswith(CB_RESUME_PAGE)
        ]
        assert page_buttons == expected_nav

    @pytest.mark.parametrize(
        ("extra", "expected_prefix", "expected_suffix"),
        [
            ({"mtime": _NOW}, "today · ", " · beef"),
            ({}, "never · ", " · beef"),
            ({"mtime": "not-a-number"}, "never · ", " · beef"),
            ({"mtime": _NOW, "msg_count": 42}, "today · ", " · 42 msgs"),
            ({"mtime": _NOW, "msg_count": 0}, "today · ", " · beef"),
        ],
        ids=[
            "known-mtime",
            "missing-mtime",
            "unparsable-mtime",
            "msg-count",
            "zero-msg-count",
        ],
    )
    def test_button_label(
        self, extra: dict, expected_prefix: str, expected_suffix: str
    ) -> None:
        sessions = [
            {
                "session_id": "abcd1234-eeee-ffff-0000-1111deadbeef",
                "summary": "Implement auth",
                "cwd": "/proj/a",
                **extra,
            }
        ]

        with patch(f"{_RC}.time.time", return_value=_NOW):
            kb = _build_resume_keyboard(sessions)

        label = kb.inline_keyboard[1][0].text
        assert label.startswith(expected_prefix)
        assert label.endswith(expected_suffix)
        assert "Implement auth" in label


# ── /resume command ───────────────────────────────────────────────────────


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
        user_data: dict = {}

        await resume_command(_make_update(), _make_context(user_data))

        mock_safe_reply.assert_called_once()
        assert "Select a session" in mock_safe_reply.call_args.args[1]
        assert len(user_data[RESUME_SESSIONS]) == 2

    @patch(f"{_RC}.scan_all_sessions", return_value=[])
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_no_sessions_shows_message(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        _mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True

        await resume_command(_make_update(), _make_context())

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

        await resume_command(_make_update(), _make_context())

        mock_safe_reply.assert_called_once()
        assert "named topic" in mock_safe_reply.call_args.args[1]

    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.get_provider")
    @patch(f"{_RC}.config")
    async def test_provider_without_resume_picker_rejected(
        self,
        mock_config: MagicMock,
        mock_get_provider: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        caps = mock_get_provider.return_value.capabilities
        caps.supports_resume = True
        caps.supports_resume_picker = False

        with patch(f"{_RC}.thread_router") as mock_tr:
            mock_tr.get_window_for_thread.return_value = None
            await resume_command(_make_update(), _make_context())

        mock_scan.assert_not_called()
        assert "not supported" in mock_safe_reply.call_args.args[1]

    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.config")
    async def test_unauthorized_user_is_ignored(
        self,
        mock_config: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = False

        await resume_command(_make_update(), _make_context())

        mock_safe_reply.assert_not_called()
        mock_scan.assert_not_called()

    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    async def test_no_message_returns_early(
        self, mock_safe_reply: AsyncMock, mock_scan: MagicMock
    ) -> None:
        update = MagicMock()
        update.message = None

        await resume_command(update, _make_context())

        mock_safe_reply.assert_not_called()
        mock_scan.assert_not_called()


# ── Resume picker callbacks ───────────────────────────────────────────────


@pytest.fixture()
def pick_env():
    """Module singletons patched for the ``rp:`` pick flow (happy path)."""
    with (
        patch(f"{_RC}.tmux_manager") as tmux,
        patch(f"{_RC}.thread_router") as router,
        patch(f"{_RC}.session_manager"),
        patch(f"{_RC}.session_map_sync") as sync,
        patch(f"{_RC}.safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_RC}.get_thread_id", return_value=42),
        patch(f"{_RC}.Path") as path,
    ):
        router.get_window_for_thread.return_value = None
        router.resolve_chat_id.return_value = -100999
        tmux.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        sync.wait_for_session_map_entry = AsyncMock()
        path.return_value.is_dir.return_value = True
        yield SimpleNamespace(tmux=tmux, router=router, sync=sync, edit=edit, path=path)


async def _pick(index: int, sessions: list[dict], ctx: MagicMock) -> AsyncMock:
    update = _make_callback_update(data=f"{CB_RESUME_PICK}{index}")
    ctx.user_data[RESUME_SESSIONS] = sessions
    query = update.callback_query
    await handle_resume_command_callback(query, 100, query.data, update, ctx)
    return query


class TestResumePickCallback:
    @pytest.mark.parametrize(
        ("index", "expected_id"), [(0, _SID), (1, _SID2)], ids=["first", "second"]
    )
    async def test_pick_creates_window_resuming_the_chosen_session(
        self, pick_env, index: int, expected_id: str
    ) -> None:
        ctx = _make_context()

        with patch(
            f"{_RC}.window_query.resolve_window_alias", return_value="@canonical"
        ) as resolve_alias:
            await _pick(
                index,
                [_session(_SID, "Fix bug"), _session(_SID2, "Add tests")],
                ctx,
            )

        pick_env.tmux.create_window.assert_called_once_with(
            "/tmp/proj",
            agent_args=f"--resume {expected_id}",
            launch_command="claude",
        )
        pick_env.sync.wait_for_session_map_entry.assert_awaited_once_with(
            "@5", timeout=5.0, resolve_window_id=resolve_alias
        )
        pick_env.router.bind_thread.assert_called_once_with(
            100, 42, "@canonical", window_name="project"
        )

    async def test_pick_unbinds_the_dead_window_first(self, pick_env) -> None:
        pick_env.router.get_window_for_thread.return_value = "@0"

        await _pick(0, [_session()], _make_context())

        pick_env.router.unbind_thread.assert_called_once_with(100, 42)

    async def test_pick_sets_group_chat_id(self, pick_env) -> None:
        await _pick(0, [_session()], _make_context())

        pick_env.router.set_group_chat_id.assert_called_once_with(100, 42, -100999)

    async def test_pick_clears_resume_state_on_success(self, pick_env) -> None:
        ctx = _make_context()

        await _pick(0, [_session()], ctx)

        assert RESUME_SESSIONS not in ctx.user_data

    async def test_pick_survives_topic_rename_failure(self, pick_env) -> None:
        with patch(f"{_RC}.PTBTelegramClient") as mock_client:
            mock_client.return_value.edit_forum_topic = AsyncMock(
                side_effect=TelegramError("topic gone")
            )
            ctx = _make_context()
            query = await _pick(0, [_session()], ctx)

        assert "Resuming session" in pick_env.edit.call_args.args[1]
        query.answer.assert_awaited_with("Resumed")
        assert RESUME_SESSIONS not in ctx.user_data

    async def test_pick_reports_window_creation_failure(self, pick_env) -> None:
        pick_env.tmux.create_window = AsyncMock(
            return_value=(False, "Tmux error", None, None)
        )
        ctx = _make_context()

        query = await _pick(0, [_session()], ctx)

        assert "Tmux error" in pick_env.edit.call_args.args[1]
        assert RESUME_SESSIONS not in ctx.user_data
        assert _toast(query) == "Couldn't create window"

    async def test_pick_invalid_cwd_fails(self, pick_env) -> None:
        pick_env.path.return_value.is_dir.return_value = False
        ctx = _make_context()

        await _pick(0, [_session(cwd="/gone")], ctx)

        pick_env.edit.assert_called_once()
        assert "no longer exists" in pick_env.edit.call_args.args[1].lower()
        assert RESUME_SESSIONS not in ctx.user_data

    @pytest.mark.parametrize(
        ("data", "sessions", "expected_toast"),
        [
            (f"{CB_RESUME_PICK}99", [_session()], "Session no longer in list"),
            (f"{CB_RESUME_PICK}-1", [_session()], "Session no longer in list"),
            (f"{CB_RESUME_PICK}0", [], "Session no longer in list"),
            (f"{CB_RESUME_PICK}notanumber", [_session()], "Couldn't read selection"),
        ],
        ids=["index-past-end", "negative-index", "no-sessions", "non-numeric-index"],
    )
    async def test_pick_guards(
        self, data: str, sessions: list[dict], expected_toast: str
    ) -> None:
        update = _make_callback_update(data=data)
        ctx = _make_context({RESUME_SESSIONS: sessions} if sessions else {})
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()
        assert _toast(query) == expected_toast

    async def test_pick_outside_a_topic_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        ctx = _make_context({RESUME_SESSIONS: [_session()]})
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=None):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert _toast(query) == "Use in a topic"


class TestResumePageCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_page_shows_sessions(self, mock_safe_edit: AsyncMock) -> None:
        sessions = [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(10)
        ]
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}1")
        query = update.callback_query

        await handle_resume_command_callback(
            query, 100, query.data, update, _make_context({RESUME_SESSIONS: sessions})
        )

        mock_safe_edit.assert_called_once()
        assert "Select a session" in mock_safe_edit.call_args.args[1]
        keyboard = mock_safe_edit.call_args.kwargs["reply_markup"]
        picks = [
            b.callback_data
            for row in keyboard.inline_keyboard
            for b in row
            if isinstance(b.callback_data, str)
            and b.callback_data.startswith(CB_RESUME_PICK)
        ]
        assert picks == [f"{CB_RESUME_PICK}{i}" for i in range(6, 10)]

    @pytest.mark.parametrize(
        ("data", "user_data", "expected_toast"),
        [
            (f"{CB_RESUME_PAGE}abc", {}, "Invalid page"),
            (f"{CB_RESUME_PAGE}0", {}, "No sessions available"),
        ],
        ids=["non-numeric-page", "no-stored-sessions"],
    )
    async def test_page_guards(
        self, data: str, user_data: dict, expected_toast: str
    ) -> None:
        update = _make_callback_update(data=data)
        query = update.callback_query

        await handle_resume_command_callback(
            query, 100, query.data, update, _make_context(user_data)
        )

        query.answer.assert_called_once()
        assert _toast(query) == expected_toast


class TestResumeCancelCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_cancel_clears_state_and_answers(
        self, mock_safe_edit: AsyncMock
    ) -> None:
        user_data: dict = {RESUME_SESSIONS: [_session()]}
        update = _make_callback_update(data=CB_RESUME_CANCEL)
        query = update.callback_query

        await handle_resume_command_callback(
            query, 100, query.data, update, _make_context(user_data)
        )

        assert RESUME_SESSIONS not in user_data
        assert "cancelled" in mock_safe_edit.call_args.args[1].lower()
        query.answer.assert_called_once_with("Cancelled")


class TestResumePerWindowProvider:
    @patch(f"{_RC}.get_provider_for_window")
    async def test_pick_uses_per_window_provider_when_bound(
        self, mock_gpw: MagicMock, pick_env
    ) -> None:
        pick_env.router.get_window_for_thread.return_value = "@3"
        mock_gpw.return_value.make_launch_args.return_value = "--resume sess-1"

        await _pick(0, [_session("sess-1")], _make_context())

        mock_gpw.assert_called_once_with("@3", provider_name=ANY)

    @patch(f"{_RC}.get_provider_for_window")
    async def test_pick_uses_provider_recorded_on_the_session(
        self, mock_gpw: MagicMock, pick_env
    ) -> None:
        mock_gpw.return_value.make_launch_args.return_value = "resume sess-1"

        await _pick(0, [_session("sess-1", provider_name="codex")], _make_context())

        mock_gpw.assert_called_once_with("", provider_name="codex")

    @patch(f"{_RC}.get_provider")
    async def test_pick_falls_back_to_global_provider_when_unbound(
        self, mock_gp: MagicMock, pick_env
    ) -> None:
        mock_gp.return_value.make_launch_args.return_value = "--resume sess-1"

        await _pick(0, [_session("sess-1")], _make_context())

        mock_gp.assert_called_once()
        mock_gp.return_value.make_launch_args.assert_called_once_with(
            resume_id="sess-1"
        )
