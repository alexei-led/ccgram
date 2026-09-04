from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_DIR_CANCEL,
    CB_MODE_SELECT,
    CB_PROV_SELECT,
)
from ccgram.handlers.topics.directory_browser import (
    build_mode_picker,
    build_provider_picker,
)
from ccgram.handlers.topics.directory_callbacks import (
    _accept_yolo_confirmation,
    _handle_confirm,
    _handle_mode_select,
    _handle_provider_select,
)
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT
from ccgram.window_query import resolve_window_alias


def _buttons(keyboard: InlineKeyboardMarkup) -> list:
    return [btn for row in keyboard.inline_keyboard for btn in row]


class TestBuildProviderPicker:
    def test_header_names_the_selected_directory(self) -> None:
        text, keyboard = build_provider_picker("/home/user/my-project")
        assert "Select Provider" in text
        assert "my-project" in text
        assert isinstance(keyboard, InlineKeyboardMarkup)

    def test_home_directory_is_shown_as_tilde(self) -> None:
        text, _keyboard = build_provider_picker(f"{Path.home()}/project")
        assert "~/project" in text

    @pytest.mark.parametrize(
        ("provider", "label"),
        [
            ("claude", "Claude"),
            ("codex", "Codex"),
            ("gemini", "Gemini"),
            ("pi", "Pi"),
            ("shell", "Shell"),
        ],
    )
    def test_offers_every_provider(self, provider: str, label: str) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        buttons = _buttons(keyboard)
        assert any(label in btn.text for btn in buttons)
        assert f"{CB_PROV_SELECT}{provider}" in [btn.callback_data for btn in buttons]

    def test_claude_marked_as_default(self) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        claude_labels = [btn.text for btn in _buttons(keyboard) if "Claude" in btn.text]
        assert any("default" in label for label in claude_labels)

    def test_has_cancel_button(self) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        assert CB_DIR_CANCEL in [btn.callback_data for btn in _buttons(keyboard)]


class TestBuildModePicker:
    def test_offers_normal_yolo_and_cancel_for_the_chosen_provider(self) -> None:
        text, keyboard = build_mode_picker("/tmp/test", "codex")
        assert "Select Session Mode" in text
        callbacks = [btn.callback_data for btn in _buttons(keyboard)]
        assert f"{CB_MODE_SELECT}codex:normal" in callbacks
        assert f"{CB_MODE_SELECT}codex:yolo" in callbacks
        assert CB_DIR_CANCEL in callbacks


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    return ctx


def _make_query(
    data: str = "", *, chat_type: str = "supergroup", chat_id: int = -100999
) -> AsyncMock:
    query = AsyncMock()
    query.data = data
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.chat.type = chat_type
    query.message.chat.id = chat_id
    return query


def _make_update(thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 100
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    update.callback_query.message.chat.type = "supergroup"
    update.callback_query.message.chat.id = -100999
    return update


class TestHandleConfirmShowsProviderPicker:
    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.workspace_callbacks.tmux_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_confirm_shows_provider_picker(
        self, mock_tr: MagicMock, mock_mux: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_mux.capabilities.native_agent_status = False
        user_data = {
            "browse_path": "/tmp/test",
            PENDING_THREAD_ID: 42,
        }
        query = _make_query()
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_confirm(query, 100, update, context)

        mock_edit.assert_called_once()
        call_args = mock_edit.call_args
        text = call_args[0][1]
        assert "Select Provider" in text
        keyboard = call_args.kwargs.get("reply_markup") or call_args[0][2]
        assert isinstance(keyboard, InlineKeyboardMarkup)

    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.workspace_callbacks.tmux_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_confirm_keeps_browse_state_for_the_provider_step(
        self, mock_tr: MagicMock, mock_mux: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_mux.capabilities.native_agent_status = False
        user_data = {
            "browse_path": "/tmp/test",
            "browse_page": 2,
            "browse_dirs": ["a", "b"],
            "state": "browsing_directory",
            PENDING_THREAD_ID: 42,
        }
        query = _make_query()
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_confirm(query, 100, update, context)

        assert "browse_path" in user_data
        assert "state" in user_data


class TestHandleProviderSelect:
    @patch(
        "ccgram.handlers.topics.provider_mode_callbacks.safe_edit",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.thread_router")
    async def test_shows_mode_picker(
        self,
        mock_tr: MagicMock,
        mock_registry: MagicMock,
        mock_tmux: MagicMock,
        mock_edit: AsyncMock,
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_tr.get_window_for_thread.return_value = None
        mock_tmux.create_window = AsyncMock()

        user_data = {"browse_path": "/tmp/test", PENDING_THREAD_ID: 42}
        query = _make_query(data=f"{CB_PROV_SELECT}codex")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_provider_select(
            query, 100, f"{CB_PROV_SELECT}codex", update, context
        )

        mock_tmux.create_window.assert_not_called()
        mock_edit.assert_called_once()
        text = mock_edit.call_args[0][1]
        assert "Select Session Mode" in text

    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    async def test_rejects_unknown_provider(self, mock_registry: MagicMock) -> None:
        mock_registry.is_valid.return_value = False
        query = _make_query(data=f"{CB_PROV_SELECT}unknown")
        update = _make_update()
        context = _make_context()

        await _handle_provider_select(
            query, 100, f"{CB_PROV_SELECT}unknown", update, context
        )
        query.answer.assert_any_call("Unknown provider", show_alert=True)

    @patch(
        "ccgram.handlers.topics.provider_mode_callbacks.safe_edit",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    async def test_state_lost_fails_closed(
        self, mock_registry: MagicMock, mock_tmux: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_tmux.create_window = AsyncMock()
        query = _make_query(data=f"{CB_PROV_SELECT}claude")
        update = _make_update()
        context = _make_context({})

        await _handle_provider_select(
            query, 100, f"{CB_PROV_SELECT}claude", update, context
        )

        mock_tmux.create_window.assert_not_called()
        assert "expired" in mock_edit.call_args[0][1].lower()


class TestHandleModeSelect:
    @patch("ccgram.providers.resolve_launch_command")
    @patch(
        "ccgram.handlers.topics.window_launch_service.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.window_launch_service.provider_registry")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    @patch("ccgram.handlers.topics.window_launch_service.thread_router")
    async def test_creates_window_with_yolo_mode(
        self,
        mock_tr: MagicMock,
        mock_registry: MagicMock,
        mock_registry_wls: MagicMock,
        mock_tmux: MagicMock,
        mock_sm: MagicMock,
        mock_edit: AsyncMock,
        mock_resolve_launch: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.has_yolo_confirmation = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = mock_provider
        mock_registry_wls.get.return_value = mock_provider

        mock_resolve_launch.return_value = (
            "codex --dangerously-bypass-approvals-and-sandbox"
        )
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'proj'", "proj", "@5")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_tmux.capabilities.native_worktrees = False
        mock_tmux.capabilities.native_agent_status = False
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = 123
        mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
        mock_sm.get_window_state.return_value = MagicMock()

        user_data = {"browse_path": "/tmp/proj", PENDING_THREAD_ID: 42}
        query = _make_query(data=f"{CB_MODE_SELECT}codex:yolo")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}codex:yolo", update, context
        )

        mock_resolve_launch.assert_called_once_with("codex", approval_mode="yolo")
        mock_tmux.create_window.assert_called_once_with(
            "/tmp/proj",
            launch_command="codex --dangerously-bypass-approvals-and-sandbox",
            workspace_id=None,
        )
        mock_tmux.stamp_pane_title.assert_awaited_once_with("@5", "codex")
        mock_sm.set_window_provider.assert_called_once_with("@5", "codex")
        mock_sm.set_window_approval_mode.assert_called_once_with("@5", "yolo")
        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -100999)

    @patch(
        "ccgram.handlers.topics.window_launch_service._accept_yolo_confirmation",
        new_callable=AsyncMock,
    )
    @patch("ccgram.providers.resolve_launch_command")
    @patch(
        "ccgram.handlers.topics.window_launch_service.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.window_launch_service.session_map_sync")
    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.window_launch_service.provider_registry")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    @patch("ccgram.handlers.topics.window_launch_service.thread_router")
    async def test_claude_yolo_accepts_bypass_prompt(
        self,
        mock_tr: MagicMock,
        mock_registry: MagicMock,
        mock_registry_wls: MagicMock,
        mock_tmux: MagicMock,
        mock_sm: MagicMock,
        mock_sms: MagicMock,
        mock_edit: AsyncMock,
        mock_resolve_launch: MagicMock,
        mock_accept_yolo: AsyncMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        mock_provider.capabilities.has_yolo_confirmation = True
        mock_provider.capabilities.chat_first_command_path = False
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = mock_provider
        mock_registry_wls.get.return_value = mock_provider

        mock_resolve_launch.return_value = "claude --dangerously-skip-permissions"
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'proj'", "proj", "@5")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_tmux.capabilities.native_worktrees = False
        mock_tmux.capabilities.native_agent_status = False
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = 123
        mock_sm.get_window_state.return_value = MagicMock()
        mock_sms.wait_for_session_map_entry = AsyncMock(return_value=True)
        mock_accept_yolo.return_value = True

        user_data = {"browse_path": "/tmp/proj", PENDING_THREAD_ID: 42}
        query = _make_query(data=f"{CB_MODE_SELECT}claude:yolo")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}claude:yolo", update, context
        )

        mock_accept_yolo.assert_awaited_once_with("@5")
        mock_sms.wait_for_session_map_entry.assert_awaited_once_with(
            "@5", resolve_window_id=resolve_window_alias
        )

    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    async def test_rejects_unknown_mode(self, mock_registry: MagicMock) -> None:
        mock_registry.is_valid.return_value = True
        query = _make_query(data=f"{CB_MODE_SELECT}codex:unknown")
        update = _make_update()
        context = _make_context()

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}codex:unknown", update, context
        )
        query.answer.assert_any_call("Unknown mode", show_alert=True)

    @patch(
        "ccgram.handlers.topics.provider_mode_callbacks.safe_edit",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    async def test_state_lost_fails_closed(
        self, mock_registry: MagicMock, mock_tmux: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_tmux.create_window = AsyncMock()
        query = _make_query(data=f"{CB_MODE_SELECT}codex:normal")
        update = _make_update()
        context = _make_context({})

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}codex:normal", update, context
        )

        mock_tmux.create_window.assert_not_called()
        assert "expired" in mock_edit.call_args[0][1].lower()

    @patch("ccgram.providers.resolve_launch_command")
    @patch(
        "ccgram.handlers.topics.window_launch_service.safe_edit", new_callable=AsyncMock
    )
    @patch(
        "ccgram.handlers.topics.window_launch_service.send_telegram_to_window",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    @patch("ccgram.handlers.topics.window_launch_service.provider_registry")
    @patch("ccgram.handlers.topics.provider_mode_callbacks.provider_registry")
    @patch("ccgram.handlers.topics.window_launch_service.thread_router")
    async def test_forwards_pending_text(
        self,
        mock_tr: MagicMock,
        mock_registry: MagicMock,
        mock_registry_wls: MagicMock,
        mock_tmux: MagicMock,
        mock_sm: MagicMock,
        mock_send_to_window: AsyncMock,
        mock_edit: AsyncMock,
        mock_resolve_launch: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = mock_provider
        mock_registry_wls.get.return_value = mock_provider

        mock_resolve_launch.return_value = "claude"
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'proj'", "proj", "@1")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_tmux.capabilities.native_worktrees = False
        mock_tmux.capabilities.native_agent_status = False
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = 123
        mock_send_to_window.return_value = (True, "ok")
        mock_sm.get_window_state.return_value = MagicMock()

        user_data = {
            "browse_path": "/tmp/proj",
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "hello world",
        }
        query = _make_query(data=f"{CB_MODE_SELECT}claude:normal")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}claude:normal", update, context
        )

        mock_send_to_window.assert_called_once_with(100, "@1", 42, "hello world", ANY)
        assert PENDING_THREAD_TEXT not in user_data


class TestAcceptYoloConfirmation:
    @pytest.fixture(autouse=True)
    def _instant_yolo_sleep(self, monkeypatch):
        from ccgram.handlers.topics import window_launch_service

        monkeypatch.setattr(window_launch_service.asyncio, "sleep", AsyncMock())

    @pytest.mark.parametrize(
        "pane_text",
        [
            "WARNING: Claude Code running in Bypass Permissions mode\n"
            "  ❯ 1. No, exit\n"
            "    2. Yes, I accept all responsibility",
            "BYPASS PERMISSIONS mode warning",
        ],
        ids=["full_prompt", "uppercase"],
    )
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    async def test_detected_prompt_is_answered_with_down_then_enter(
        self, mock_tmux: MagicMock, pane_text: str
    ) -> None:
        from unittest.mock import call

        mock_tmux.capture_pane = AsyncMock(return_value=pane_text)
        mock_tmux.send_keys = AsyncMock(return_value=True)

        assert await _accept_yolo_confirmation("@5", timeout=2.0) is True
        assert mock_tmux.send_keys.call_args_list == [
            call("@5", "Down", enter=False, literal=False),
            call("@5", "Enter", enter=False, literal=False),
        ]

    @pytest.mark.parametrize(
        "capture", [None, "some other output"], ids=["no_capture", "unrelated_output"]
    )
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    async def test_timeout_without_prompt_sends_no_keys(
        self, mock_tmux: MagicMock, capture: str | None
    ) -> None:
        mock_tmux.capture_pane = AsyncMock(return_value=capture)
        mock_tmux.send_keys = AsyncMock(return_value=True)

        assert await _accept_yolo_confirmation("@5", timeout=0.1) is False
        mock_tmux.send_keys.assert_not_awaited()

    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    async def test_polls_until_prompt_appears(self, mock_tmux: MagicMock) -> None:
        mock_tmux.capture_pane = AsyncMock(
            side_effect=[
                None,
                "Loading...",
                "Bypass Permissions mode\n❯ 1. No, exit",
            ]
        )
        mock_tmux.send_keys = AsyncMock(return_value=True)

        assert await _accept_yolo_confirmation("@5", timeout=5.0) is True
        assert mock_tmux.capture_pane.await_count == 3

    @patch("ccgram.handlers.topics.window_launch_service.asyncio.get_running_loop")
    @patch("ccgram.handlers.topics.window_launch_service.tmux_manager")
    async def test_configured_timeout_allows_slow_cold_start(
        self, mock_tmux: MagicMock, mock_get_loop: MagicMock, monkeypatch
    ) -> None:
        mock_get_loop.return_value.time.side_effect = [0.0, 0.0, 9.0, 9.0]
        mock_tmux.capture_pane = AsyncMock(
            side_effect=["Loading...", "Bypass Permissions mode\n❯ 1. No, exit"]
        )
        mock_tmux.send_keys = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "ccgram.handlers.topics.window_launch_service.config.yolo_confirmation_timeout",
            30.0,
            raising=False,
        )

        assert await _accept_yolo_confirmation("@5") is True
        assert mock_tmux.capture_pane.await_count == 2
