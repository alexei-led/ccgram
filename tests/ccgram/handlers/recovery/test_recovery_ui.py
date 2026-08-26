import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

import ccgram.handlers.commands.menu_sync as cmd_orch_mod
from ccgram.handlers.text.text_handler import text_handler
from ccgram.handlers.recovery.recovery_banner import (
    _recovery_help_text,
    build_recovery_keyboard,
)
from ccgram.handlers.recovery.recovery_callbacks import handle_recovery_callback
from ccgram.handlers.recovery.resume_command import ResumeEntry
from ccgram.handlers.recovery.resume_picker import (
    _SessionEntry,
    _build_empty_resume_keyboard,
    scan_sessions_for_cwd,
)
from ccgram.handlers.callback_data import (
    CB_RECOVERY_BACK,
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_RESUME,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    RECOVERY_SESSIONS,
    RESUME_SESSIONS,
    RECOVERY_WINDOW_ID,
)

# After Round 5 split: _handle_fresh/continue/resume/back/cancel/browse,
# _create_and_bind_window, build_recovery_keyboard, _recovery_help_text,
# scan_sessions_for_cwd (re-imported here), session_manager, thread_router,
# tmux_manager, etc. all live in recovery_banner. Keeping the constant name
# minimises diff against pre-split tests.
_RC = "ccgram.handlers.recovery.recovery_banner"
_RP = "ccgram.handlers.recovery.resume_picker"
_TH = "ccgram.handlers.text.text_handler"

_STALE_TOAST = "Stale recovery (topic mismatch)"


@pytest.fixture(autouse=True)
def _patch_recovery_session_manager():
    """Mock session_manager writes (set_window_*) so tests don't hit real state."""
    with patch(f"{_RC}.session_manager"):
        yield


def _make_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
    thread_id: int = 42,
    text: str = "hello",
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


def _recovery_user_data(
    thread_id: int = 42,
    text: str = "hello",
    window_id: str = "@0",
) -> dict:
    return {
        PENDING_THREAD_ID: thread_id,
        PENDING_THREAD_TEXT: text,
        RECOVERY_WINDOW_ID: window_id,
    }


async def _tap(data: str, ctx: MagicMock, *, thread_id: int = 42) -> AsyncMock:
    """Dispatch one recovery button tap and return its CallbackQuery."""
    update = _make_callback_update(data=data, thread_id=thread_id)
    query = update.callback_query
    await handle_recovery_callback(query, 100, query.data, update, ctx)
    return query


def _toast(query: AsyncMock) -> str:
    call = query.answer.call_args
    assert call is not None
    return str(call.args[0]) if call.args else str(call.kwargs.get("text", ""))


def _capabilities(
    mock_gpw: MagicMock, *, cont: bool, resume: bool, picker: bool = True
) -> None:
    caps = mock_gpw.return_value.capabilities
    caps.supports_continue = cont
    caps.supports_resume = resume
    caps.supports_resume_picker = picker


def _callback_datas(keyboard) -> list[str]:
    return [
        b.callback_data
        for row in keyboard.inline_keyboard
        for b in row
        if isinstance(b.callback_data, str)
    ]


# ── Recovery keyboard + help text ─────────────────────────────────────────


class TestBuildRecoveryKeyboard:
    def test_action_row_then_cancel_row(self) -> None:
        kb = build_recovery_keyboard("@5")

        assert [b.callback_data for b in kb.inline_keyboard[0]] == [
            f"{CB_RECOVERY_FRESH}@5",
            f"{CB_RECOVERY_CONTINUE}@5",
            f"{CB_RECOVERY_RESUME}@5",
        ]
        assert [b.callback_data for b in kb.inline_keyboard[1]] == [CB_RECOVERY_CANCEL]

    @pytest.mark.parametrize(
        ("cont", "resume", "picker", "expected_prefixes"),
        [
            (
                True,
                True,
                True,
                [CB_RECOVERY_FRESH, CB_RECOVERY_CONTINUE, CB_RECOVERY_RESUME],
            ),
            (False, True, True, [CB_RECOVERY_FRESH, CB_RECOVERY_RESUME]),
            (True, False, True, [CB_RECOVERY_FRESH, CB_RECOVERY_CONTINUE]),
            (True, True, False, [CB_RECOVERY_FRESH, CB_RECOVERY_CONTINUE]),
            (False, False, True, [CB_RECOVERY_FRESH]),
        ],
        ids=["full", "no-continue", "no-resume", "no-picker", "fresh-only"],
    )
    def test_actions_follow_provider_capabilities(
        self, cont: bool, resume: bool, picker: bool, expected_prefixes: list[str]
    ) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            _capabilities(mock_gpw, cont=cont, resume=resume, picker=picker)
            kb = build_recovery_keyboard("@0")

        datas = [str(b.callback_data) for b in kb.inline_keyboard[0]]
        assert [d[: len(p)] for d, p in zip(datas, expected_prefixes)] == (
            expected_prefixes
        )
        assert len(datas) == len(expected_prefixes)

    def test_uses_per_window_provider(self) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            _capabilities(mock_gpw, cont=True, resume=True)
            build_recovery_keyboard("@7")

        mock_gpw.assert_called_once_with("@7", provider_name=None)


class TestRecoveryHelpText:
    @pytest.mark.parametrize(
        ("cont", "resume", "picker", "expected"),
        [
            (
                True,
                True,
                True,
                "Start fresh · Continue last session · Resume from list",
            ),
            (False, True, True, "Start fresh · Resume from list"),
            (True, False, True, "Start fresh · Continue last session"),
            (True, True, False, "Start fresh · Continue last session"),
            (False, False, True, "Start fresh"),
        ],
        ids=["full", "no-continue", "no-resume", "no-picker", "fresh-only"],
    )
    def test_help_text_follows_provider_capabilities(
        self, cont: bool, resume: bool, picker: bool, expected: str
    ) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            _capabilities(mock_gpw, cont=cont, resume=resume, picker=picker)
            assert _recovery_help_text("@0") == expected

    def test_uses_per_window_provider(self) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            _capabilities(mock_gpw, cont=True, resume=True)
            _recovery_help_text("@9")

        mock_gpw.assert_called_once_with("@9", provider_name=None)


# ── Text handler entry into recovery ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _allow_user():
    with patch(f"{_TH}.config.is_user_allowed", return_value=True):
        yield


@pytest.fixture()
def _no_group():
    with patch(f"{_TH}.config") as mock_config:
        mock_config.group_id = None
        mock_config.is_user_allowed = MagicMock(return_value=True)
        yield mock_config


@pytest.fixture()
def dead_window(_no_group):
    """A bound topic whose window is gone, with the text handler's deps mocked."""
    with (
        patch(f"{_TH}.thread_router") as router,
        patch(f"{_TH}.tmux_manager") as tmux,
        patch(f"{_TH}.window_query") as wq,
        patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as reply,
        patch(f"{_TH}.build_directory_browser") as browser,
        patch(f"{_TH}.Path") as path,
    ):
        router.get_window_for_thread.return_value = "@0"
        router.get_display_name.return_value = "project"
        tmux.find_window_by_id = AsyncMock(return_value=None)
        wq.view_window.return_value = MagicMock(cwd="/tmp/project")
        browser.return_value = ("Browse:", MagicMock(), [])
        path.return_value.is_dir.return_value = True
        path.cwd.return_value = path.return_value
        path.cwd.return_value.__str__ = MagicMock(return_value="/cwd")
        yield SimpleNamespace(
            router=router, tmux=tmux, wq=wq, reply=reply, browser=browser, path=path
        )


class TestTextHandlerDeadWindow:
    async def test_dead_window_shows_recovery_ui(self, dead_window) -> None:
        await text_handler(_make_update(), _make_context())

        dead_window.reply.assert_called_once()
        msg_text = dead_window.reply.call_args.args[1]
        assert "ended" in msg_text
        assert "recover" in msg_text.lower()

    async def test_dead_window_stores_pending_message(self, dead_window) -> None:
        user_data: dict = {}

        await text_handler(
            _make_update(text="my pending message"), _make_context(user_data)
        )

        assert user_data[PENDING_THREAD_TEXT] == "my pending message"
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[RECOVERY_WINDOW_ID] == "@0"

    async def test_dead_window_keeps_the_binding(self, dead_window) -> None:
        await text_handler(_make_update(), _make_context())

        dead_window.router.unbind_thread.assert_not_called()

    @pytest.mark.parametrize(
        "cwd", ["", "/nonexistent/path"], ids=["no-cwd", "cwd-gone"]
    )
    async def test_unusable_cwd_unbinds_and_falls_back_to_browser(
        self, dead_window, cwd: str
    ) -> None:
        dead_window.wq.view_window.return_value = MagicMock(cwd=cwd)
        dead_window.path.return_value.is_dir.return_value = False

        await text_handler(_make_update(), _make_context())

        dead_window.router.unbind_thread.assert_called_once()
        dead_window.browser.assert_called_once()


class TestBotTextHandlerScopedMenu:
    @patch(f"{_TH}.handle_text_message", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.commands.menu_sync.sync_scoped_provider_menu",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.commands.menu_sync.get_provider_for_window")
    @patch("ccgram.handlers.commands.menu_sync.thread_router")
    async def test_syncs_scoped_menu_when_thread_is_bound(
        self,
        mock_tr: MagicMock,
        mock_get_provider: MagicMock,
        mock_sync_menu: AsyncMock,
        mock_handle_text: AsyncMock,
        _no_group: MagicMock,
    ) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        mock_get_provider.return_value = provider
        mock_tr.resolve_window_for_thread.return_value = "@1"

        update = _make_update()
        ctx = _make_context()

        await text_handler(update, ctx)

        mock_sync_menu.assert_called_once_with(update.message, 100, provider)
        mock_handle_text.assert_called_once_with(update, ctx)

    @patch(f"{_TH}.handle_text_message", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.commands.menu_sync.sync_scoped_provider_menu",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.commands.menu_sync.thread_router")
    async def test_skips_scoped_menu_sync_when_thread_is_unbound(
        self,
        mock_tr: MagicMock,
        mock_sync_menu: AsyncMock,
        mock_handle_text: AsyncMock,
        _no_group: MagicMock,
    ) -> None:
        mock_tr.resolve_window_for_thread.return_value = None

        update = _make_update()
        ctx = _make_context()

        await text_handler(update, ctx)

        mock_sync_menu.assert_not_called()
        mock_handle_text.assert_called_once_with(update, ctx)

    @patch(f"{_TH}.handle_text_message", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.commands.menu_sync.sync_scoped_provider_menu",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.commands.menu_sync.get_provider_for_window")
    @patch("ccgram.handlers.commands.menu_sync.thread_router")
    async def test_cached_chat_user_still_resolves_provider_context(
        self,
        mock_tr: MagicMock,
        mock_get_provider: MagicMock,
        mock_sync_menu: AsyncMock,
        mock_handle_text: AsyncMock,
        _no_group: MagicMock,
    ) -> None:
        cmd_orch_mod._scoped_provider_menu.clear()
        try:
            cmd_orch_mod._scoped_provider_menu[(-100999, 100)] = "codex"
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
            mock_get_provider.return_value = provider
            mock_tr.resolve_window_for_thread.return_value = "@1"
            update = _make_update()
            update.message.chat.id = -100999
            ctx = _make_context()

            await text_handler(update, ctx)

            mock_tr.resolve_window_for_thread.assert_called_once_with(100, 42, -100999)
            mock_sync_menu.assert_called_once_with(update.message, 100, provider)
            mock_handle_text.assert_called_once_with(update, ctx)
        finally:
            cmd_orch_mod._scoped_provider_menu.clear()


# ── Recovery button handlers ──────────────────────────────────────────────


@pytest.fixture()
def recovery_env():
    """Module singletons patched for the recovery-banner button handlers."""
    with (
        patch(f"{_RC}.thread_router") as router,
        patch(f"{_RC}.tmux_manager") as tmux,
        patch(f"{_RC}.window_query") as wq,
        patch(f"{_RC}.session_map_sync") as sync,
        patch(f"{_RC}.safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_RC}.safe_send", new_callable=AsyncMock) as send,
        patch(f"{_RC}.send_telegram_to_window", new_callable=AsyncMock) as forward,
        patch(f"{_RC}.Path") as path,
        patch(f"{_RP}.window_query") as picker_wq,
        patch(f"{_RP}.Path") as picker_path,
    ):
        view = MagicMock(cwd="/tmp/project", provider_name="")
        wq.view_window.return_value = view
        wq.resolve_window_alias.side_effect = lambda window_id: window_id
        picker_wq.view_window.return_value = view
        tmux.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        sync.wait_for_session_map_entry = AsyncMock()
        forward.return_value = (True, "ok")
        router.resolve_chat_id.return_value = -100999
        path.return_value.is_dir.return_value = True
        picker_path.return_value.is_dir.return_value = True
        yield SimpleNamespace(
            router=router,
            tmux=tmux,
            wq=wq,
            sync=sync,
            edit=edit,
            send=send,
            forward=forward,
            path=path,
            picker_wq=picker_wq,
            picker_path=picker_path,
        )


class TestRecoveryFreshCallback:
    async def test_fresh_creates_window_and_rebinds(self, recovery_env) -> None:
        recovery_env.wq.resolve_window_alias.side_effect = None
        recovery_env.wq.resolve_window_alias.return_value = "@canonical"

        await _tap(f"{CB_RECOVERY_FRESH}@0", _make_context(_recovery_user_data()))

        recovery_env.router.unbind_thread.assert_called_once_with(100, 42)
        recovery_env.tmux.create_window.assert_called_once_with(
            "/tmp/project", agent_args="", launch_command="claude"
        )
        recovery_env.sync.wait_for_session_map_entry.assert_awaited_once_with(
            "@5", timeout=5.0, resolve_window_id=recovery_env.wq.resolve_window_alias
        )
        recovery_env.router.bind_thread.assert_called_once_with(
            100, 42, "@canonical", window_name="project", chat_id=-100999
        )
        recovery_env.router.set_group_chat_id.assert_called_once_with(100, 42, -100999)

    async def test_fresh_forwards_pending_message_and_clears_state(
        self, recovery_env
    ) -> None:
        user_data = _recovery_user_data()

        await _tap(f"{CB_RECOVERY_FRESH}@0", _make_context(user_data))

        recovery_env.forward.assert_called_once_with(100, "@5", 42, "hello", -100999)
        assert PENDING_THREAD_TEXT not in user_data
        assert PENDING_THREAD_ID not in user_data
        assert RECOVERY_WINDOW_ID not in user_data

    async def test_fresh_reports_a_failed_pending_forward(self, recovery_env) -> None:
        recovery_env.forward.return_value = (False, "pane gone")

        query = await _tap(
            f"{CB_RECOVERY_FRESH}@0", _make_context(_recovery_user_data())
        )

        recovery_env.send.assert_awaited_once()
        assert "Failed to send pending message" in recovery_env.send.call_args.args[2]
        assert _toast(query) == "Created"

    async def test_fresh_reports_window_creation_failure(self, recovery_env) -> None:
        recovery_env.tmux.create_window = AsyncMock(
            return_value=(False, "no free pane", None, None)
        )
        user_data = _recovery_user_data()

        query = await _tap(f"{CB_RECOVERY_FRESH}@0", _make_context(user_data))

        assert "no free pane" in recovery_env.edit.call_args.args[1]
        assert _toast(query) == "Failed"
        assert user_data == {}
        recovery_env.router.bind_thread.assert_not_called()


class TestRecoveryContinueCallback:
    @patch(f"{_RC}.scan_sessions_for_cwd", return_value=[_SessionEntry("s1", "x")])
    async def test_continue_probes_the_provider_it_will_actually_launch(
        self, mock_scan: MagicMock, recovery_env
    ) -> None:
        """Continue launches exactly one provider, so it must probe that one.

        Resume may widen an unknown provider because each entry carries its own
        to the relaunch; probing wider here would find another agent's
        sessions, skip the empty state, and run `<default> --continue` against
        a folder it has nothing to continue.
        """
        recovery_env.wq.get_window_provider.return_value = ""

        await _tap(f"{CB_RECOVERY_CONTINUE}@0", _make_context(_recovery_user_data()))

        assert mock_scan.call_args.args[1]

    @patch(f"{_RC}.scan_sessions_for_cwd", return_value=[_SessionEntry("s1", "x")])
    async def test_continue_creates_window_with_continue_flag(
        self, _mock_scan: MagicMock, recovery_env
    ) -> None:
        await _tap(f"{CB_RECOVERY_CONTINUE}@0", _make_context(_recovery_user_data()))

        recovery_env.tmux.create_window.assert_called_once_with(
            "/tmp/project", agent_args="--continue", launch_command="claude"
        )
        recovery_env.router.bind_thread.assert_called_once_with(
            100, 42, "@5", window_name="project", chat_id=-100999
        )

    @patch(f"{_RC}.scan_sessions_for_cwd", return_value=[_SessionEntry("s1", "x")])
    async def test_continue_forwards_pending_message(
        self, _mock_scan: MagicMock, recovery_env
    ) -> None:
        user_data = _recovery_user_data(text="my message")

        await _tap(f"{CB_RECOVERY_CONTINUE}@0", _make_context(user_data))

        recovery_env.forward.assert_called_once_with(
            100, "@5", 42, "my message", -100999
        )
        assert PENDING_THREAD_TEXT not in user_data


class TestRecoveryResumeCallback:
    @patch(f"{_RC}.scan_sessions_for_cwd")
    async def test_resume_shows_session_picker(
        self, mock_scan: MagicMock, recovery_env
    ) -> None:
        mock_scan.return_value = [
            _SessionEntry("sess-1", "Fix login bug"),
            _SessionEntry("sess-2", "Add tests"),
        ]
        user_data = _recovery_user_data()

        await _tap(f"{CB_RECOVERY_RESUME}@0", _make_context(user_data))

        recovery_env.edit.assert_called_once()
        assert "Select a session" in recovery_env.edit.call_args.args[1]
        assert [s["session_id"] for s in user_data[RECOVERY_SESSIONS]] == [
            "sess-1",
            "sess-2",
        ]

    @patch(f"{_RC}.scan_sessions_for_cwd")
    async def test_picker_stores_mtime_for_pick_callback(
        self, mock_scan: MagicMock, recovery_env
    ) -> None:
        mock_scan.return_value = [_SessionEntry("sess-1", "X", 12345.0)]
        user_data = _recovery_user_data()

        await _tap(f"{CB_RECOVERY_RESUME}@0", _make_context(user_data))

        assert user_data[RECOVERY_SESSIONS][0]["mtime"] == 12345.0

    @pytest.mark.parametrize(
        ("mtime", "expected_prefix"),
        [(None, "today · "), (0.0, "never · ")],
        ids=["recent", "unknown-mtime"],
    )
    @patch(f"{_RC}.scan_sessions_for_cwd")
    async def test_picker_labels_use_the_shared_formatter(
        self,
        mock_scan: MagicMock,
        recovery_env,
        mtime: float | None,
        expected_prefix: str,
    ) -> None:
        import time

        mock_scan.return_value = [
            _SessionEntry(
                "a1b2c3-0000-1111-2222-3333deadbeef",
                "Fix login bug",
                time.time() - 10 if mtime is None else mtime,
            )
        ]

        await _tap(f"{CB_RECOVERY_RESUME}@0", _make_context(_recovery_user_data()))

        keyboard = recovery_env.edit.call_args.kwargs["reply_markup"]
        label = keyboard.inline_keyboard[0][0].text
        assert label.startswith(expected_prefix)
        assert "Fix login bug" in label
        assert label.endswith(" · beef")


class TestRecoveryResumePickCallback:
    @pytest.mark.parametrize(
        ("index", "expected_id"),
        [
            (0, "a1b2c3d4-0000-0000-0000-000000000001"),
            (1, "a1b2c3d4-0000-0000-0000-000000000002"),
        ],
        ids=["first", "second"],
    )
    async def test_pick_creates_window_with_resume_flag(
        self, recovery_env, index: int, expected_id: str
    ) -> None:
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [
            {
                "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                "summary": "Fix login bug",
            },
            {
                "session_id": "a1b2c3d4-0000-0000-0000-000000000002",
                "summary": "Add tests",
            },
        ]

        await _tap(f"{CB_RECOVERY_PICK}{index}", _make_context(user_data))

        recovery_env.tmux.create_window.assert_called_once_with(
            "/tmp/project",
            agent_args=f"--resume {expected_id}",
            launch_command="claude",
        )
        recovery_env.router.bind_thread.assert_called_once()

    @pytest.mark.parametrize(
        ("data", "sessions", "expected_toast"),
        [
            (
                f"{CB_RECOVERY_PICK}99",
                [{"session_id": "x", "summary": "x"}],
                "Session no longer in list",
            ),
            (f"{CB_RECOVERY_PICK}0", None, "Session no longer in list"),
            (
                f"{CB_RECOVERY_PICK}notanumber",
                [{"session_id": "x", "summary": "x"}],
                "Couldn't read selection",
            ),
        ],
        ids=["index-past-end", "no-sessions-stored", "non-numeric-index"],
    )
    async def test_pick_guards(
        self, data: str, sessions: list | None, expected_toast: str
    ) -> None:
        user_data = _recovery_user_data()
        if sessions is not None:
            user_data[RECOVERY_SESSIONS] = sessions

        query = await _tap(data, _make_context(user_data))

        query.answer.assert_called_once()
        assert _toast(query) == expected_toast

    async def test_pick_outside_a_topic_rejected(self) -> None:
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [{"session_id": "x", "summary": "x"}]

        with patch(f"{_RP}.get_thread_id", return_value=None):
            query = await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        assert _toast(query) == "Use in a topic"

    async def test_pick_without_a_remembered_window_rejected(self) -> None:
        user_data = {
            PENDING_THREAD_ID: 42,
            RECOVERY_SESSIONS: [{"session_id": "x", "summary": "x"}],
        }

        query = await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        assert _toast(query) == "Recovery menu expired"

    @pytest.mark.parametrize(
        "view", [None, MagicMock(cwd="")], ids=["no-window-state", "no-cwd"]
    )
    async def test_pick_reports_missing_state_not_a_missing_folder(
        self, recovery_env, view
    ) -> None:
        """Without window state the folder is unknown, not deleted (#176)."""
        recovery_env.wq.view_window.return_value = view
        recovery_env.picker_wq.view_window.return_value = view
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [{"session_id": "x", "summary": "x"}]

        query = await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        text = recovery_env.edit.call_args.args[1]
        assert "no longer exists" not in text.lower()
        assert "session state is gone" in text
        assert _toast(query) == "State gone"

    async def test_pick_launches_the_entrys_own_provider(self, recovery_env) -> None:
        """A window with no provider widens the picker; the pick must decide.

        Inheriting from the old window resolves the same falsy value to the
        config default, so a picked codex session would launch claude with
        codex-format resume arguments.
        """
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [
            {"session_id": "x", "summary": "Codex work", "provider_name": "codex"}
        ]

        await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        launch_command = recovery_env.tmux.create_window.call_args.kwargs[
            "launch_command"
        ]
        assert "codex" in launch_command, launch_command

    async def test_pick_rejects_a_session_from_another_provider(
        self, recovery_env
    ) -> None:
        recovery_env.picker_wq.view_window.return_value = MagicMock(
            cwd="/tmp/project", provider_name="claude"
        )
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [
            {"session_id": "x", "summary": "x", "provider_name": "codex"}
        ]

        query = await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        assert _toast(query) == "Session provider mismatch"
        recovery_env.tmux.create_window.assert_not_called()


class TestRecoveryBackCallback:
    async def test_back_shows_recovery_menu_with_help_text(self, recovery_env) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            _capabilities(mock_gpw, cont=True, resume=True)
            query = await _tap(
                f"{CB_RECOVERY_BACK}@0", _make_context(_recovery_user_data())
            )

        recovery_env.edit.assert_called_once()
        body = recovery_env.edit.call_args.args[1]
        assert "Choose how to continue" in body
        assert "Start fresh · Continue last session · Resume from list" in body
        query.answer.assert_called_once()


class TestRecoveryCancelCallback:
    async def test_cancel_clears_all_recovery_state(self, recovery_env) -> None:
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [{"session_id": "x", "summary": "y"}]

        await _tap(CB_RECOVERY_CANCEL, _make_context(user_data))

        assert user_data == {}
        recovery_env.edit.assert_called_once()


class TestRecoveryCallbackDispatch:
    async def test_unknown_recovery_data_is_acknowledged(self) -> None:
        query = await _tap("rec:zzz", _make_context({}))

        query.answer.assert_awaited_once_with()

    async def test_expired_token_reports_and_stops(self) -> None:
        from ccgram.handlers.recovery.recovery_callbacks import _dispatch

        update = _make_callback_update(data=f"{CB_RECOVERY_FRESH}token")
        update.message = None
        query = update.callback_query

        with (
            patch(
                "ccgram.handlers.recovery.recovery_callbacks.resolve_callback_data",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.recovery.recovery_callbacks.handle_recovery_callback",
                new_callable=AsyncMock,
            ) as handler,
        ):
            await _dispatch(update, _make_context({}))

        handler.assert_not_awaited()
        query.answer.assert_awaited_once_with(
            "This button has expired", show_alert=True
        )


class TestRecoveryStaleGuards:
    """Every recovery button refuses a tap that no longer matches its state."""

    _BUTTONS = [
        f"{CB_RECOVERY_FRESH}@0",
        f"{CB_RECOVERY_CONTINUE}@0",
        f"{CB_RECOVERY_RESUME}@0",
        f"{CB_RECOVERY_BACK}@0",
        f"{CB_RECOVERY_BROWSE}@0",
        f"{CB_RECOVERY_PICK}0",
        CB_RECOVERY_CANCEL,
    ]

    @pytest.mark.parametrize("data", _BUTTONS)
    async def test_tap_from_another_topic_rejected(self, data: str) -> None:
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [{"session_id": "x", "summary": "x"}]

        query = await _tap(data, _make_context(user_data), thread_id=99)

        query.answer.assert_called_once()
        assert _toast(query) == _STALE_TOAST

    @pytest.mark.parametrize(
        "data",
        [
            f"{CB_RECOVERY_FRESH}@0",
            f"{CB_RECOVERY_CONTINUE}@0",
            f"{CB_RECOVERY_RESUME}@0",
            f"{CB_RECOVERY_BACK}@0",
            f"{CB_RECOVERY_BROWSE}@0",
        ],
    )
    @patch(f"{_RC}.thread_router")
    async def test_tap_without_pending_state_or_binding_rejected(
        self, mock_router: MagicMock, data: str
    ) -> None:
        mock_router.get_window_for_thread.return_value = None

        query = await _tap(data, _make_context({}))

        query.answer.assert_called_once()
        assert _toast(query) == _STALE_TOAST

    @pytest.mark.parametrize(
        "data",
        [
            f"{CB_RECOVERY_FRESH}@999",
            f"{CB_RECOVERY_CONTINUE}@999",
            f"{CB_RECOVERY_RESUME}@999",
            f"{CB_RECOVERY_BACK}@999",
            f"{CB_RECOVERY_BROWSE}@999",
        ],
    )
    async def test_tap_for_a_different_window_rejected(self, data: str) -> None:
        user_data = {PENDING_THREAD_ID: 42, RECOVERY_WINDOW_ID: "@0"}

        query = await _tap(data, _make_context(user_data))

        query.answer.assert_called_once()
        assert _toast(query) == _STALE_TOAST

    @pytest.mark.parametrize(
        "data",
        [f"{CB_RECOVERY_FRESH}@0", f"{CB_RECOVERY_CONTINUE}@0"],
        ids=["fresh", "continue"],
    )
    async def test_launch_fails_when_cwd_is_gone(self, recovery_env, data: str) -> None:
        recovery_env.path.return_value.is_dir.return_value = False
        recovery_env.wq.view_window.return_value = MagicMock(cwd="/gone")

        query = await _tap(data, _make_context(_recovery_user_data()))

        recovery_env.edit.assert_called_once()
        assert "no longer exists" in recovery_env.edit.call_args.args[1].lower()
        assert _toast(query) == "Project gone"


# ── Empty state + cross-project browse ────────────────────────────────────


class TestRecoveryEmptyStateAndBrowseFallback:
    @pytest.mark.parametrize(
        "data",
        [f"{CB_RECOVERY_RESUME}@0", f"{CB_RECOVERY_CONTINUE}@0"],
        ids=["resume", "continue"],
    )
    @patch(f"{_RC}.scan_sessions_for_cwd", return_value=[])
    async def test_no_sessions_offers_browse_and_fresh(
        self, _mock_scan: MagicMock, recovery_env, data: str
    ) -> None:
        await _tap(data, _make_context(_recovery_user_data()))

        recovery_env.edit.assert_called_once()
        assert "No sessions" in recovery_env.edit.call_args.args[1]
        keyboard = recovery_env.edit.call_args.kwargs["reply_markup"]
        datas = _callback_datas(keyboard)
        assert any(d.startswith(CB_RECOVERY_BROWSE) for d in datas)
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert CB_RECOVERY_CANCEL in datas

    @patch("ccgram.handlers.recovery.resume_command.scan_all_sessions")
    async def test_browse_loads_cross_project_picker(
        self, mock_scan_all: MagicMock, recovery_env
    ) -> None:
        mock_scan_all.return_value = [
            ResumeEntry("sess-x", "From other project", "/other", 12345.0),
        ]
        user_data = _recovery_user_data(text="pending text")

        await _tap(f"{CB_RECOVERY_BROWSE}@0", _make_context(user_data))

        assert user_data[RESUME_SESSIONS][0]["session_id"] == "sess-x"
        assert PENDING_THREAD_TEXT not in user_data
        assert "Select a session" in recovery_env.edit.call_args.args[1]

    @patch("ccgram.handlers.recovery.resume_command.scan_all_sessions", return_value=[])
    async def test_browse_with_no_sessions_anywhere(
        self, _mock_scan_all: MagicMock, recovery_env
    ) -> None:
        user_data = _recovery_user_data()

        query = await _tap(f"{CB_RECOVERY_BROWSE}@0", _make_context(user_data))

        recovery_env.edit.assert_called_once()
        assert "No past sessions" in recovery_env.edit.call_args.args[1]
        assert _toast(query) == "Nothing to resume"
        assert user_data == {}


class TestEmptyStateKeyboardBuilder:
    def test_empty_keyboard_has_browse_fresh_and_cancel(self) -> None:
        datas = _callback_datas(_build_empty_resume_keyboard("@0"))

        assert any(d.startswith(CB_RECOVERY_BROWSE) for d in datas)
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert CB_RECOVERY_CANCEL in datas

    def test_empty_keyboard_callback_data_within_64_bytes(self) -> None:
        kb = _build_empty_resume_keyboard("@" + "x" * 60)

        for data in _callback_datas(kb):
            assert len(data.encode("utf-8")) <= 64


# ── Per-window provider wiring ────────────────────────────────────────────


class TestRecoveryPerWindowProvider:
    @patch(f"{_RC}.scan_sessions_for_cwd", return_value=[_SessionEntry("s1", "x")])
    @patch(f"{_RC}.get_provider_for_window")
    async def test_continue_uses_per_window_provider(
        self, mock_gpw: MagicMock, _mock_scan: MagicMock, recovery_env
    ) -> None:
        mock_gpw.return_value.make_launch_args.return_value = "--continue"

        await _tap(f"{CB_RECOVERY_CONTINUE}@0", _make_context(_recovery_user_data()))

        mock_gpw.assert_called_with("@0", provider_name=ANY)
        mock_gpw.return_value.make_launch_args.assert_called_once_with(
            use_continue=True
        )

    @patch(f"{_RP}.get_provider_for_window")
    async def test_resume_pick_uses_per_window_provider(
        self, mock_picker_gpw: MagicMock, recovery_env
    ) -> None:
        mock_picker_gpw.return_value.make_launch_args.return_value = "--resume sess-1"
        user_data = _recovery_user_data()
        user_data[RECOVERY_SESSIONS] = [
            {"session_id": "sess-1", "summary": "Fix login bug"},
        ]

        await _tap(f"{CB_RECOVERY_PICK}0", _make_context(user_data))

        mock_picker_gpw.assert_called_with("@0", provider_name=ANY)
        mock_picker_gpw.return_value.make_launch_args.assert_called_once_with(
            resume_id="sess-1"
        )


# ── Per-cwd session discovery ─────────────────────────────────────────────
#
# The provider-level scan (index parsing, summary fallbacks, mtime ordering,
# dedup) is covered by test_resume_command.py::TestScanAllSessions. These
# tests only pin the cwd filter that ``scan_sessions_for_cwd`` adds on top.


@pytest.fixture()
def projects_root(tmp_path: Path):
    root = tmp_path / "projects"
    with patch("ccgram.providers.claude._claude_projects_path", return_value=root):
        yield root


def _index_for(proj_dir: Path, cwd: str, *entries: dict) -> None:
    (proj_dir / "sessions-index.json").write_text(
        json.dumps({"originalPath": cwd, "entries": list(entries)})
    )


class TestScanSessionsForCwd:
    def test_returns_sessions_matching_cwd(self, projects_root, tmp_path) -> None:
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        proj_dir = projects_root / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        _index_for(
            proj_dir,
            str(work_dir.resolve()),
            {
                "sessionId": "sess-1",
                "fullPath": str(session_file),
                "projectPath": str(work_dir.resolve()),
                "summary": "Fix the bug",
            },
        )

        result = scan_sessions_for_cwd(str(work_dir))

        assert len(result) == 1
        assert result[0].session_id == "sess-1"
        assert result[0].summary == "Fix the bug"

    def test_indexed_session_for_another_cwd_is_filtered_out(
        self, projects_root, tmp_path
    ) -> None:
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        proj_dir = projects_root / "-tmp-other"
        proj_dir.mkdir(parents=True)
        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        _index_for(
            proj_dir,
            str(other_dir.resolve()),
            {
                "sessionId": "sess-1",
                "fullPath": str(session_file),
                "projectPath": str(other_dir.resolve()),
            },
        )

        assert scan_sessions_for_cwd(str(work_dir)) == []

    def test_bare_jsonl_matching_cwd_is_returned(self, projects_root, tmp_path) -> None:
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        proj_dir = projects_root / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "abc-123.jsonl").write_text(
            json.dumps(
                {
                    "type": "user",
                    "cwd": str(work_dir.resolve()),
                    "message": {"content": [{"type": "text", "text": "Fix bug"}]},
                }
            )
            + "\n"
        )

        result = scan_sessions_for_cwd(str(work_dir))

        assert len(result) == 1
        assert result[0].session_id == "abc-123"
        assert result[0].summary == "Fix bug"

    def test_bare_jsonl_for_another_cwd_is_filtered_out(
        self, projects_root, tmp_path
    ) -> None:
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        proj_dir = projects_root / "-tmp-other"
        proj_dir.mkdir(parents=True)
        (proj_dir / "abc-123.jsonl").write_text(
            json.dumps(
                {
                    "type": "user",
                    "cwd": str(other_dir.resolve()),
                    "message": {"content": "hi"},
                }
            )
            + "\n"
        )

        assert scan_sessions_for_cwd(str(work_dir)) == []

    def test_caps_the_picker_at_six_newest_sessions(
        self, projects_root, tmp_path
    ) -> None:
        import os

        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        proj_dir = projects_root / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        resolved = str(work_dir.resolve())
        entries = []
        for i in range(7):
            session_file = proj_dir / f"sess-{i}.jsonl"
            session_file.write_text('{"type":"summary"}\n')
            os.utime(session_file, (1_700_000_000 + i, 1_700_000_000 + i))
            entries.append(
                {
                    "sessionId": f"sess-{i}",
                    "fullPath": str(session_file),
                    "projectPath": resolved,
                    "summary": f"Session {i}",
                }
            )
        _index_for(proj_dir, resolved, *entries)

        result = scan_sessions_for_cwd(str(work_dir))

        assert [entry.session_id for entry in result] == [
            f"sess-{i}" for i in range(6, 0, -1)
        ]

    def test_unresolvable_cwd_returns_empty_without_scanning(
        self, projects_root
    ) -> None:
        with patch(f"{_RP}.get_provider_for_window") as mock_gpw:
            assert scan_sessions_for_cwd("/bad\x00path") == []

        mock_gpw.assert_not_called()
